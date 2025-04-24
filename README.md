# CSYE7105_FINAL

Northeastern University CSYE7105 final project

there is some sample images in samples folder

## Usage

```bash
pip install -r requirements.txt
```

## source code

ddpm/

## For inference

cifar10_inference.ipynb is for inference, need to download the checkpoint file first.

link is in report(conclusion section)

change the `cpt_path` to the path of the checkpoint file.

## For benchmark

dp_benchmark.py

ddp_benchmark.py

fsdp_benchmark.py

## For FID score

FID.ipynb

## For DDPM

ddpm.py

## For performance analysis

performance.ipynb

## For training

```python
def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPM with DDP')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--load_cpt', type=str, default=None, help='load checkpoint path')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--interval', type=int, default=25, help='Interval of saving checkpoint')
    parser.add_argument('--only_model', action='store_true', help='Only load model weights, not training state')
    
    ret = parser.parse_args()

    return ret
```

dp.py

ddp.py (there is bug in this file, since we have finished training, we have not fixed it, we will fix it in resubmission)