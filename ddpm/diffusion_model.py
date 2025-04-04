# diffusion_model.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.config import DDPMConfig
from ddpm.res_net import ResNet
from typing import Optional

def extract(a, t, x_shape):
    """
    Like your utils.extract(): gather values along t, then reshape to x_shape.
    a is shape [num_timesteps], t is shape [batch], x_shape is [batch, ...].
    """
    batch_size = t.shape[0]
    out = a.gather(dim=0, index=t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def make_linear_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """
    Example: linear schedule from beta_start to beta_end.
    """
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)

class DiffusionModel(nn.Module):
    """
    Diffusion model class that wraps:
      - A ResNet-based U-Net
      - Beta schedule / alpha-cumprod computation
      - Forward diffusion (perturb_x), reverse diffusion (sample), training losses, etc.
    """
    def __init__(self, config: DDPMConfig):
        super().__init__()
        self.config = config
        self.model = ResNet(config)

        if config.betas is None:
            betas = make_linear_schedule(
                num_timesteps=config.num_timesteps,
                beta_start=1e-4, 
                beta_end=0.02   
            )
        else:
            betas = np.array(config.betas, dtype=np.float32)
            if len(betas) != config.num_timesteps:
                raise ValueError("Length of config.betas must match config.num_timesteps")

        self.num_timesteps = config.num_timesteps
        self.register_buffer("betas", torch.tensor(betas))

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.register_buffer("alphas", torch.tensor(alphas))
        self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", torch.tensor(alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod", torch.tensor(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.tensor(np.sqrt(1 - alphas_cumprod)))

        self.register_buffer("sqrt_recip_alphas", torch.tensor(np.sqrt(1.0 / alphas)))

        # For q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_variance", 
            torch.tensor(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        )
        # The log calculation can be used for stable sampling; omitted for brevity.

        # We also might want to store self.loss_type = config.loss_type

    def perturb_x(self, x0, t, noise=None):
        """
        Forward diffusion: sample x_t from x_0
          x_t = sqrt_alphas_cumprod[t] * x_0
                + sqrt(1 - alphas_cumprod[t]) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def get_losses(self, x0, t, y=None):
        """
        1) Draw noise sample
        2) Get x_t = q(x_t|x_0)
        3) Predict noise via self.model
        4) Return MSE or L1 between predicted noise and real noise
        """
        noise = torch.randn_like(x0)
        x_noisy = self.perturb_x(x0, t, noise)
        pred_noise = self.model(x_noisy, t.float(), y)
        
        if self.config.loss_type == "l1":
            loss = F.l1_loss(pred_noise, noise)
        else:
            loss = F.mse_loss(pred_noise, noise)
        return loss

    def forward(self, x0, y=None):
        """
        Single training step:
         - Random t
         - Compute diffusion loss
        """
        b, c, h, w = x0.shape
        device = x0.device

        # pick random timesteps
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        loss = self.get_losses(x0, t, y)
        return loss

    @torch.no_grad()
    def remove_noise(self, x, t, y=None):
        """
        One reverse step: estimate x_{t-1} from x_t.
        x_{t-1} = 1/sqrt(alpha_t) * [ x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) ]
                  + some variance term (sigma_t * z)
        For simplicity, we do the "predicted x0" route or the "predict noise" route. 
        This snippet uses the typical DDPM formula for x_{t-1}.
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # model predicts noise
        eps = self.model(x, t.float(), y)

        # Equation 11 in DDPM paper:
        x = sqrt_recip_alphas_t * (x - betas_t / (1.0 - extract(self.alphas_cumprod, t, x.shape))**0.5 * eps)
        return x

    @torch.no_grad()
    def p_sample(self, x, t, y=None):
        """
        One reverse diffusion step, with the variance term included.
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # equation (11)
        eps = self.model(x, t.float(), y)
        x_model = sqrt_recip_alphas_t * (x - betas_t / (1 - extract(self.alphas_cumprod, t, x.shape))**0.5 * eps)

        # if t > 0, we add noise
        if (t > 0).any():
            # sigma = sqrt(posterior_variance_t)
            posterior_var_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            x_model = x_model + torch.sqrt(posterior_var_t) * noise
        return x_model

    @torch.no_grad()
    def sample(self, shape, device, y=None):
        """
        Sample from pure noise (x_T ~ N(0,1)) down to x_0
        shape is (batch_size, channels, height, width)
        """
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            # each i is a timestep
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, y=y)
        return x

    @torch.no_grad()
    def sample_intermediate(self, shape, device, y=None, save_steps: list[int] = []):
        """
        If you want to store the intermediate x_t as you go (for visualization).
        """
        b = shape[0]
        x = torch.randn(shape, device=device)
        xs = [x.detach().cpu()]
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, y=y)
            if i in save_steps:
                xs.append(x.detach().cpu())
        return xs
