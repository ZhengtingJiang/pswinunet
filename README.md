# Probbailistic Swin Unet

## 1. Data
You can feel free to deal with [LIDC-IDRI dataset](https://www.cancerimagingarchive.net/collection/lidc-idri/). Or you can use the preprocessed dataset in [Google Cloud](https://drive.google.com/file/d/1VZmHbnwd-XkapzrsjL9yCrnT1ERDoqw9/view?usp=sharing) a `.pickle` file. The data is cropped to a resolution of (128,128), with a fixed spacing. 
## 2. Envrionment
You should prepare an environment with python >= 3.10, and use the command for dependencies:
```
pip install -r requirements.txt
```
## 3. Train/test
We use Wandb to monitor our training progress. Details can be seen here: [wandb](https://wandb.ai/site)
