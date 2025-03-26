# train a probabilistic U-Net model

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
import pickle
import os
import wandb
import json
from torchvision.utils import make_grid
# optimization settings
lr = 1e-5
l2_reg = 1e-6
lr_decay_every = 5   # decay LR after this many epochs
lr_decay = 0.95



# checkpoint directory
out_dir = 'outputs/1'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    print('Folder already exists. Existing models and training logs will be replaced')
    
dataset = LIDC_IDRI(dataset_location = r"../data/")
dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.seed(42)
np.random.shuffle(indices)

# split
train_split = int(np.floor(0.8 * dataset_size)) # 80%
val_split = int(np.floor(0.9 * dataset_size)) # 10%

train_indices = indices[:train_split]
val_indices = indices[train_split:val_split]
test_indices = indices[val_split:] # 10%

# sampler
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# 创建 dataloader
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_config = {
    "model_name": "PSUNet",
    "backbone": "Swin-UNet",
    "input_shape": [20,1,128,128],
    "num_classes": 1,
    "filters": [16,32,64,128,256],
    "flow:": "Gaussian",
    "num_flows": 0,
    "latent_dim": 6,
    "beta": 1.0,
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "loss": "BCE + KL",
    "dataset": "LIDC-IDRI",
    "batch_size_train": 20,
    "batch_size_val": 1,
    "batch_size_test": 1,
    "seed": 42,
    "num_epochs": 250,
    "patience": 20
}
with open("model_config.json", "w") as f:
    json.dump(model_config, f)

os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
wandb.init(project="PSUNet", config=model_config)
# network
model = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[16,32,64,128,256], latent_dim=6, no_convs_fcomb=3, beta=10.0)
model.cuda()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)

# logging
train_loss = []
test_loss = []
best_val_loss = 999.0

patience = 20
epochs_no_improve = 0
early_stop = False

epochs = 250
print("Training start...")
for epoch in range(epochs):
    model.train()
    loss_train = 0
    loss_segmentation = 0
    # training loop
    for step, (patch, mask, _) in enumerate(train_loader): 
        patch = patch.cuda()
        mask = mask.cuda()
        mask = torch.unsqueeze(mask,1)
        model.forward(patch, mask, training=True)
        elbo = model.elbo(mask)
        loss = -elbo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_train += loss.detach().cpu().item()
        if step % 100 == 0:
            with torch.no_grad():
                reconstructions = model.reconstruct(use_posterior_mean=True)
                idx = 0
                img_slice = patch[idx].detach().cpu()
                seg_slice = mask[idx].detach().cpu()
                output_slice = reconstructions[idx].detach().cpu()
                output_slice = torch.sigmoid(output_slice).detach().cpu()

                img_grid = make_grid(img_slice, normalize=True)
                seg_grid = make_grid(seg_slice, normalize=False)
                pred_grid = make_grid(output_slice, normalize=False)
                binary_grid = make_grid((output_slice > 0.5).float(), normalize=False)

                wandb.log({
                    "Train_Input_Image": wandb.Image(img_grid),
                    "Train_Ground_Truth": wandb.Image(seg_grid),
                    "Train_Predicted_Image": wandb.Image(pred_grid),
                    "Train_Binary_Predicted_Mask": wandb.Image(binary_grid)
                })
    # end of training loop
    loss_train /= len(train_loader)
    
    # valdiation loop
    model.eval()
    loss_val = 0
    
    with torch.no_grad():
        for step, (patch, mask, _) in enumerate(test_loader): 
            patch = patch.cuda()
            mask = mask.cuda()
            mask = torch.unsqueeze(mask,1)
            model.forward(patch, mask, training=True)
            elbo = model.elbo(mask)
            loss = -elbo 
            
            loss_val += loss.detach().cpu().item()
            if step % 30 == 0:
                reconstructions = model.reconstruct(use_posterior_mean=True)
                output = torch.sigmoid(reconstructions)

                idx = 0
                input_img = patch[idx].detach().cpu()  # [1, H, W]
                gt_mask = mask[idx].detach().cpu()  # [1, H, W]
                pred_mask = output[idx].detach().cpu()  # [1, H, W]
                binary_pred = (pred_mask > 0.5).float()

                img_grid = make_grid(input_img, normalize=True)
                gt_grid = make_grid(gt_mask, normalize=False)
                pred_grid = make_grid(pred_mask, normalize=False)
                binary_grid = make_grid(binary_pred, normalize=False)

                wandb.log({
                    "Val/Input Image": wandb.Image(img_grid),
                    "Val/Ground Truth": wandb.Image(gt_grid),
                    "Val/Predicted Mask": wandb.Image(pred_grid),
                    "Val/Binary Predicted Mask": wandb.Image(binary_grid),
                })
    # end of validation
    loss_val /= len(test_loader)
    
    train_loss.append(loss_train)
    test_loss.append(loss_val)

    print('End of epoch ', epoch + 1, ' , Train loss: ', loss_train, ', val loss: ', loss_val)
    scheduler.step()

    wandb.log({
        "loss_val": loss_val,
        "loss_train": loss_train,
        "epoch": epoch + 1,
        "early_stop": early_stop
    })

    if loss_val < best_val_loss:
        best_val_loss = loss_val
        epochs_no_improve = 0
        fname = model_config["model_name"]
        torch.save(model.state_dict(), fname + ".pth")
        print("model saved at epoch: ", epoch + 1)
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        early_stop = True
        break
model.load_state_dict(torch.load(f"{model_config["model_name"]}_finished.pth"))
print("Finish training")