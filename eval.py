import os
import sys
import glob
import time
from typing import re
import re
import cv2
import torch
from tqdm import tqdm
from torch import einsum
import dataLoader as DLr
from utils import consistency
from network import MyNetwork
from utils.cuda_chk import GPUorCPU
from torch.utils.data import DataLoader


class Fusion:
    def __init__(self,
                 modelpath='./best_network.pth',  
                 dataroot='./Datasets/Eval',  
                 dataset_name='Lytro',  
                 threshold=0.0015,  
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.SAVEPATH = '/' + self.DATASET_NAME
        self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME == 'Lytro' or 'MFFW' or 'MFI':
            MODEL = self.LoadWeights(self.MODELPATH)
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            print("Test Dataset required!")
            pass

    def LoadWeights(self, modelpath):
        model = MyNetwork().to(self.DEVICE)
        model.load_state_dict(torch.load(modelpath, map_location=torch.device(self.DEVICE), weights_only=True))
        model.eval()
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        return model
        num_params = sum(p.numel() for p in model.parameters())
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))  # 待测源图像A
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))  # 待测源图像B
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = consistency.Binarization(img_tensor)  # 二值化
        if threshold != 0:
            Verified_img_tensor = consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results' + savepath):
            os.mkdir('./Results' + savepath)

        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)

        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()
                NetOut = model(A, B)
                D = self.ConsisVerif(NetOut, threshold)[0]
                name_A = os.path.basename(eval_list_A[cnt - 1])
                name_B = os.path.basename(eval_list_B[cnt - 1])
                print(f"Pair {cnt}: A = {name_A}, B = {name_B}")

                D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()
                A_img = cv2.imread(eval_list_A[cnt - 1])
                B_img = cv2.imread(eval_list_B[cnt - 1])
                Final_fused = A_img * D + B_img * (1 - D)

                src_name = os.path.splitext(name_A)[0]  
                match = re.search(r'\d+', src_name)
                num_str = match.group() if match else src_name  

                cv2.imwrite('./Results' + savepath + '/lytro-' + num_str + '-dm.png', D * 255)
                cv2.imwrite('./Results' + savepath + '/lytro-' + num_str + '.png', Final_fused)

                running_time.append(time.time() - start_time)
                cnt += 1

        for t in running_time:
            print("process_time: {} s".format(t))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()
