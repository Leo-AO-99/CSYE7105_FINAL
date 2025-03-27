from ddpm.data import get_lsun_church_dataloader

loader = get_lsun_church_dataloader(batch_size=16)

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

for images in loader:
    print(images.shape)
    # 显示前5张图片
    plt.figure(figsize=(15, 5))
    grid_img = vutils.make_grid(images[:5], nrow=5, normalize=True)
    plt.imshow(np.transpose(grid_img.cpu().numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()
    break