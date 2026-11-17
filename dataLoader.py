import torch
from torchvision import transforms
from torch.utils.data import Dataset
from utils.cuda_chk import GPUorCPU
from torchvision.io import read_image, ImageReadMode


DEVICE = GPUorCPU().DEVICE
model_input_image_size_height = 256
model_input_image_size_width = 256
random_crop_size = 224

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)

class DataLoader_Train(Dataset):
    train_valid_transforms = transforms.Compose(
        [
            transforms.Resize((model_input_image_size_height, model_input_image_size_width), antialias=False),  # 调整大小
            transforms.RandomCrop(random_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ZeroOneNormalize(),
        ]
    )
    train_valid_transforms_Norm = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # 数据标准化，即均值为0，标准差为1
        ]
    )

    def __init__(self, file_list_A, file_list_B, file_list_GT):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.file_list_GT = file_list_GT
        self.transform1 = self.train_valid_transforms
        self.transform2 = self.train_valid_transforms_Norm

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B) == len(self.file_list_GT):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        seed = torch.random.seed()

        imgA_path = self.file_list_A[idx]
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_A = self.transform1(img_A)
        imgA_transformed = self.transform2(img_A)

        imgB_path = self.file_list_B[idx]
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        torch.random.manual_seed(seed)
        img_B = self.transform1(img_B)
        imgB_transformed = self.transform2(img_B)

        imgGT_path = self.file_list_GT[idx]
        img_GT = read_image(imgGT_path, mode=ImageReadMode.GRAY).to(DEVICE)
        torch.random.manual_seed(seed)
        imgGT_transformed = self.transform1(img_GT)

        return imgA_transformed, imgB_transformed, imgGT_transformed


class Dataloader_Eval(Dataset):
    eval_transforms = transforms.Compose(
        [
            ZeroOneNormalize(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, file_list_A, file_list_B):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.transform1 = self.eval_transforms
        self.transform2 = self.eval_transforms

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        imgA_path = self.file_list_A[idx]
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        imgA_transformed = self.transform1(img_A)

        imgB_path = self.file_list_B[idx]
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        imgB_transformed = self.transform1(img_B)

        return imgA_transformed, imgB_transformed
