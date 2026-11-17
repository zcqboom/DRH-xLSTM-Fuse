import os
import glob
import sys
import datetime

import numpy as np
import torch
from torch import nn, Tensor
from tqdm import tqdm
import dataLoader as DLr
from network import MyNetwork
from utils.cuda_chk import GPUorCPU
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchmetrics.functional import structural_similarity_index_measure as ssim


class NetTrain:
    def __init__(self,
                 data_path='./Datasets/Train&Valid/MFFdatasets',    
                 set_size=10000,   
                 batchsize=8,   
                 epochs=66, 
                 lr=0.0005,         
                 gamma=0.88,      
                 scheduler_step=1, 
                 patience=5      
                 ):

        self.DEVICE = GPUorCPU().DEVICE     
        self.DATAPATH = data_path
        self.SETSIEZE = set_size
        self.BATCHSIZE = batchsize
        self.EPOCHS = epochs
        self.LR = lr
        self.GAMMA = gamma
        self.SCHEDULER_STEP = scheduler_step
        self.PATIENCE = patience
        self.criterion = JointLoss(gamma=0.7)

    def __call__(self, *args, **kwargs):
        TRAIN_LOADER, VALID_LOADER = self.PrepareDataLoader(self.DATAPATH, self.SETSIEZE, self.BATCHSIZE)
        MODEL, OPTIMIZER, SCHEDULER = self.BuildModel(self.DEVICE, self.LR, self.SCHEDULER_STEP, self.GAMMA)
        self.TrainingProcess(MODEL, OPTIMIZER, SCHEDULER, TRAIN_LOADER, VALID_LOADER, self.EPOCHS)

    def PrepareDataLoader(self, datapath, setsize, batchsize):
        train_list_A = sorted(glob.glob(os.path.join(datapath, 'train/sourceA', '*.*')))[:setsize]
        train_list_B = sorted(glob.glob(os.path.join(datapath, 'train/sourceB', '*.*')))[:setsize]
        train_list_GT = sorted(glob.glob(os.path.join(datapath, 'train/decisionmap', '*.*')))[:setsize]
        valid_list_A = sorted(glob.glob(os.path.join(datapath, 'validate/sourceA', '*.*')))[:setsize // 9]
        valid_list_B = sorted(glob.glob(os.path.join(datapath, 'validate/sourceB', '*.*')))[:setsize // 9]
        valid_list_GT = sorted(glob.glob(os.path.join(datapath, 'validate/decisionmap', '*.*')))[:setsize // 9]
        tqdm.write(f"Train Data A: {len(train_list_A)}")
        tqdm.write(f"Train Data B: {len(train_list_B)}")
        tqdm.write(f"Train Data GT: {len(train_list_GT)}\n")
        tqdm.write(f"Valid Data A: {len(valid_list_A)}")
        tqdm.write(f"Valid Data B: {len(valid_list_B)}")
        tqdm.write(f"Valid Data GT: {len(valid_list_GT)}\n")
        train_data = DLr.DataLoader_Train(train_list_A, train_list_B, train_list_GT)
        valid_data = DLr.DataLoader_Train(valid_list_A, valid_list_B, valid_list_GT)
        train_loader = DataLoader(dataset=train_data,       
                                  batch_size=batchsize,    
                                  shuffle=True,             
                                  num_workers=0,           
                                  pin_memory=False)         
        valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        tqdm.write(f"Train Data Size:{len(train_data)} , Train Loader Amount: {len(train_data)}/{batchsize} = {len(train_loader)}")
        tqdm.write(f"Valid Data Size:{len(valid_data)} , Valid Loader Amount: {len(valid_data)}/{batchsize} = {len(valid_loader)}\n")
        return train_loader, valid_loader

    def BuildModel(self, device, lr, scheduler_step, gamma):
        model = MyNetwork().to(device)  
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)  
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=gamma)    
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        return model, optimizer, scheduler

    def LossFunc(self, GT, NetOut):
        loss_abs = nn.L1Loss()
        loss = loss_abs(GT, NetOut)
        return loss

    def TrainingProcess(self, model, optimizer, scheduler, train_loader, valid_loader, epochs):
        torch.backends.cudnn.benchmark = True  
        scaler = torch.cuda.amp.GradScaler()    
        early_stopping = EarlyStopping('./')   

        tqdm.write('Training start...\n')

        for epoch in range(epochs):
            epoch_loss = 0      
            epoch_accuracy = 0  
            train_loader_tqdm = tqdm(train_loader, colour='green', leave=False, file=sys.stdout)    
            for A, B, GT in train_loader_tqdm:  
                optimizer.zero_grad()   
                with torch.autocast(device_type=self.DEVICE, dtype=torch.bfloat16):  
                    NetOut = model(A, B)    
                    loss = self.LossFunc(GT, NetOut)    
                scaler.scale(loss).backward() 
                scaler.step(optimizer)       
                scaler.update()         

                epoch_loss += loss / len(train_loader) 
                epoch_accuracy += (1 - loss) / len(train_loader)    
                # 配置进度条，显示实时loss等数据
                train_loader_tqdm.set_description("[%s] Epoch %s" % (str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')), str(epoch + 1)))
                train_loader_tqdm.set_postfix(loss=float(loss), acc=1 - float(loss))
          
            with torch.no_grad():       
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                valid_loader_tqdm = tqdm(valid_loader, colour='yellow', leave=False, file=sys.stdout)
                for A, B, GT in valid_loader_tqdm:
                    NetOut = model(A, B)
                    loss = self.LossFunc(GT, NetOut)
                    epoch_val_accuracy += (1 - loss) / len(valid_loader)
                    epoch_val_loss += loss / len(valid_loader)
                    valid_loader_tqdm.set_description("[Validating...] Epoch %s" % str(epoch + 1))
                    valid_loader_tqdm.set_postfix(loss=float(loss), acc=1 - float(loss))

            tqdm.write(f"[{str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S'))}] Epoch {epoch + 1} - "
                       f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - "
                       f"val_acc: {epoch_val_accuracy:.4f}")
            scheduler.step()    
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break  


if __name__ == '__main__':
    t = NetTrain()
    t()
