# -*- coding: utf-8 -*-
import os
import time
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data_generator import DataGenerator
from model_unet import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seed_all, create_dir, computation_time
print(torch.cuda.is_available())

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss= 0.0
    
    model.train()
    
    for x, y in loader:
        x=x.to(device, dtype=torch.float32)
        y=y.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        y_pred=model(x)
        loss=loss_fn(y_pred, y)
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        
    epoch_loss = epoch_loss / len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss= 0.0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x=x.to(device, dtype=torch.float32)
            y=y.to(device, dtype=torch.float32)
            
            y_pred=model(x)
            loss=loss_fn(y_pred, y)
            epoch_loss += loss.item()
            
        epoch_loss=epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """seeding"""
    seed_all()
    
    """log output"""
    create_dir('files')
    
    """load dataset"""
    train_x=sorted(glob.glob("new_data/train/image/*"))
    train_y=sorted(glob.glob("new_data/train/mask/*"))
    
    valid_x=sorted(glob.glob("new_data/test/image/*"))
    valid_y=sorted(glob.glob("new_data/test/mask/*"))
    
    
    print(f"Dataset size: \nTrain:{len(train_x)} - Vaid: {len(valid_x)}")
    
    """Hyperparameters"""
    H=256
    W=256
    size=(H, W)
    batch_size= 1
    num_epochs= 50
    lr=1e-4
    checkpoint_path="files/checkpoint.pth"
    
    
    """Dataset and DataLoader"""
    train_dataset=DataGenerator(train_x, train_y)
    valid_dataset=DataGenerator(valid_x, valid_y)
    
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
        )
    
    valid_loader=DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
        )
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model=build_unet()
    model=model.to(device)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn=DiceBCELoss()
    
    
    """Training the model"""
    best_valid_loss=float("inf")
    
    for epoch in range(num_epochs):
        start_time=time.time()
        train_loss=train(model, train_loader, optimizer, loss_fn, device)
        valid_loss=evaluate(model, valid_loader, loss_fn, device)
        
        """saving the checkpoint"""
        if valid_loss <best_valid_loss:
            print(f"Validaton loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving the checkpoint:{checkpoint_path}" )
            
            best_valid_loss=valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            
        end_time=time.time()
        epoch_mins, epoch_secs = computation_time(start_time, end_time)
        print(f"Epoch: {epoch+1:02} | time: {epoch_mins}m {epoch_secs}s \n Train loss: {train_loss:.3f} -- Val. loss: {valid_loss:.3f}")
        
        
            
            
        
    
    
        