# import os
# import sys
# import glob
# import time
# from typing import re
# import re
# import cv2
# import numpy as np
# import torch
#
# from tqdm import tqdm
# from torch import einsum
# import dataLoader as DLr
# from utils import consistency
# # from XLSTM_exp import MyNetwork
# from network import MyNetwork
# # from network_hgnn_test import MyNetwork
# # from only_Freq import MyNetwork
# from utils.cuda_chk import GPUorCPU
# from torch.utils.data import DataLoader
#
#
# class Fusion:
#     def __init__(self,
#                  modelpath='./seed/42/best_network.pth',  # 训练好的权重位置
#                  dataroot='./Datasets/Eval',  # 测试数据集根目录
#                  dataset_name='LMIF',  # 数据集名称
#                  # threshold=0.005,                # 小区域移除阈值
#                  threshold=0.0015,  # 小区域移除阈值
#                  ):
#         self.DEVICE = GPUorCPU().DEVICE
#         self.MODELPATH = modelpath
#         self.DATAROOT = dataroot
#         self.DATASET_NAME = dataset_name
#         self.THRESHOLD = threshold
#         self.SAVEPATH = '/' + self.DATASET_NAME
#         self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
#
#     def __call__(self, *args, **kwargs):
#         if self.DATASET_NAME == 'Lytro' or 'MFFW' or 'MFI' or 'Grayscale' or 'Savic' or 'HBU-CVMDSP' or 'HRMFF'or'LowLight':
#             # 加载已训练好的权重
#             MODEL = self.LoadWeights(self.MODELPATH)
#             # 准备数据
#             EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
#             # 图像融合
#             self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
#         else:
#             # 为正确选择测试数据集
#             print("Test Dataset required!")
#             pass
#
#     # 加载已训练好的模型权重
#     def LoadWeights(self, modelpath):
#         # 实例化模型到计算设备
#         model = MyNetwork().to(self.DEVICE)
#         # model = MyNetwork(graph_mode="rule_sim",
#         #                   interaction_mode="self").to(self.DEVICE)
#         # 总文件加载权重
#         model.load_state_dict(torch.load(modelpath, map_location=torch.device(self.DEVICE), weights_only=True))
#         # 将模型转为评估模式
#         model.eval()
#         # model.enable_frequency_visualization('./frequency_visualizations', every=1)
#         # 打印模型相关信息
#         num_params = 0
#         for p in model.parameters():
#             num_params += p.numel()
#         # print(model)
#         print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
#         return model
#
#         # def LoadWeights(self, modelpath):
#         #     # 实例化原始模型（不加 DataParallel）
#         #     model = MyNetwork().to(self.DEVICE)
#         #
#         #     # 加载权重（允许 weights_only=True）
#         #     state_dict = torch.load(modelpath, map_location=torch.device(self.DEVICE), weights_only=True)
#         #
#         #     # 处理多GPU训练时产生的 'module.' 前缀
#         #     new_state_dict = {}
#         #     for k, v in state_dict.items():
#         #         if k.startswith('module.'):
#         #             new_state_dict[k[7:]] = v  # 去掉 'module.'
#         #         else:
#         #             new_state_dict[k] = v
#         #
#         #     # 加载处理后的权重
#         #     model.load_state_dict(new_state_dict)
#         #     model.eval()
#
#         # 打印参数量（与原代码相同）
#         num_params = sum(p.numel() for p in model.parameters())
#         print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
#         return model
#
#     # 准备数据
#     def PrepareData(self, datapath):
#         eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))  # 待测源图像A
#         eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))  # 待测源图像B
#         # eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceD', '*.*')))   # 待测源图像A
#         # eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceC', '*.*')))   # 待测源图像B
#         return eval_list_A, eval_list_B
#
#     # 一致性检验
#     def ConsisVerif(self, img_tensor, threshold):
#         Verified_img_tensor = consistency.Binarization(img_tensor)  # 二值化
#         if threshold != 0:
#             # 小区域移除
#             Verified_img_tensor = consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
#         return Verified_img_tensor
#
#     # def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
#     #     if not os.path.exists('./Results' + savepath):
#     #         # 结果输出目录
#     #         os.mkdir('./Results' + savepath)
#     #     # 配置DataLoader
#     #     eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
#     #     eval_loader = DataLoader(dataset=eval_data,  # 已配置的Dataloader
#     #                              batch_size=1,  # 批大小
#     #                              shuffle=False)  # 是否随机洗牌（评估时建议关闭）
#     #     # 实例化tqdm对象
#     #     eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
#     #     cnt = 1
#     #     running_time = []
#     #     with torch.no_grad():
#     #         for A, B in eval_loader_tqdm:
#     #             # 记录开始时间
#     #             start_time = time.time()
#     #             # 源图像喂模型，进行预测
#     #             NetOut = model(A, B)
#     #             # 对网络的预测结果进行一致性检验得到融合决策图D
#     #             D = self.ConsisVerif(NetOut, threshold)[0]
#     #
#     #             # ---------- 打印当前处理的图像对名字 ----------
#     #             name_A = os.path.basename(eval_list_A[cnt - 1])
#     #             name_B = os.path.basename(eval_list_B[cnt - 1])
#     #             print(f"Pair {cnt}: A='{name_A}', B='{name_B}'")
#     #
#     #             #################################### Image fusion #######################################
#     #             D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()  # 转为opencv支持的whc模式
#     #             A = cv2.imread(eval_list_A[cnt - 1])  # 用opencv打开源图像A
#     #             B = cv2.imread(eval_list_B[cnt - 1])  # 用opencv打开源图像B
#     #             Final_fused = A * D + B * (1 - D)  # 图像融合
#     #             # 写回聚焦决策图和融合图像
#     #             cv2.imwrite('./Results' + savepath + '/lytro-' + str(cnt).zfill(2) + '-dm.png', D * 255)
#     #             cv2.imwrite('./Results' + savepath + '/lytro-' + str(cnt).zfill(2) + '.png', Final_fused)
#     #             #########################################################################################
#     #             # 记录每张图的融合时间
#     #             running_time.append(time.time() - start_time)
#     #             cnt += 1
#     #
#     #     # 打印融合时间
#     #     for i in range(len(running_time)):
#     #         print("process_time: {} s".format(running_time[i]))
#     #     print("\nResults are saved in: " + "./Results" + savepath)
#     def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
#         if not os.path.exists('./Results' + savepath):
#             os.mkdir('./Results' + savepath)
#
#         eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
#         eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
#         eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
#
#         cnt = 1
#         running_time = []
#         with torch.no_grad():
#             for A, B in eval_loader_tqdm:
#                 start_time = time.time()
#                 NetOut = model(A, B)
#                 D = self.ConsisVerif(NetOut, threshold)[0]
#
#                 # ----- 打印当前处理的文件名，方便检查匹配 -----
#                 name_A = os.path.basename(eval_list_A[cnt - 1])
#                 name_B = os.path.basename(eval_list_B[cnt - 1])
#                 print(f"Pair {cnt}: A = {name_A}, B = {name_B}")
#
#                 D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()
#                 A_img = cv2.imread(eval_list_A[cnt - 1])
#                 B_img = cv2.imread(eval_list_B[cnt - 1])
#                 Final_fused = A_img * D + B_img * (1 - D)
#
#                 # ----- 提取源文件名中的数字，用于保存 -----
#                 src_name = os.path.splitext(name_A)[0]  # 去掉扩展名
#                 match = re.search(r'\d+', src_name)
#                 num_str = match.group() if match else src_name  # 如果没数字就用整个文件名
#
#                 cv2.imwrite('./Results' + savepath + '/lytro-' + num_str + '-dm.png', D * 255)
#                 cv2.imwrite('./Results' + savepath + '/lytro-' + num_str + '.png', Final_fused)
#
#                 running_time.append(time.time() - start_time)
#                 cnt += 1
#
#         for t in running_time:
#             print("process_time: {} s".format(t))
#         print("\nResults are saved in: " + "./Results" + savepath)
#
#
# if __name__ == '__main__':
#     f = Fusion()
#     f()


# # 决策图变好了，但是指标降低了
# import os
# import sys
# import glob
# import time
#
# import cv2
# import numpy as np
# import torch
# from torch import einsum
# from tqdm import tqdm
#
# import dataLoader as DLr
# from utils import consistency
# from network import MyNetwork
# from utils.cuda_chk import GPUorCPU
# from torch.utils.data import DataLoader
#
#
# class Fusion:
#     def __init__(self,
#                  modelpath='./best_network_be.pth',
#                  dataroot='./Datasets/Eval',
#                  dataset_name='Lytro',
#                  threshold=0.0015):
#         self.DEVICE = GPUorCPU().DEVICE
#         self.MODELPATH = modelpath
#         self.DATAROOT = dataroot
#         self.DATASET_NAME = dataset_name
#         self.THRESHOLD = threshold
#         self.SAVEPATH = '/' + self.DATASET_NAME
#         self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
#
#         # 导向滤波参数（可调节）
#         self.guided_radius = 4
#         self.guided_eps = 1e-6
#
#     def __call__(self, *args, **kwargs):
#         if self.DATASET_NAME in ['Lytro', 'MFFW', 'MFI', 'Grayscale', 'Savic', 'HBU-CVMDSP', 'HRMFF']:
#             MODEL = self.LoadWeights(self.MODELPATH)
#             EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
#             self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
#         else:
#             print("Test Dataset required!")
#
#     def LoadWeights(self, modelpath):
#         model = MyNetwork().to(self.DEVICE)
#         state_dict = torch.load(modelpath, map_location=torch.device(self.DEVICE), weights_only=True)
#
#         # 兼容多GPU训练保存的权重（移除 'module.' 前缀）
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 new_state_dict[k[7:]] = v
#             else:
#                 new_state_dict[k] = v
#         model.load_state_dict(new_state_dict)
#         model.eval()
#
#         num_params = sum(p.numel() for p in model.parameters())
#         print("The number of model parameters: {} M\n\n".format(round(num_params / 1e6, 6)))
#         return model
#
#     def PrepareData(self, datapath):
#         eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
#         eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
#         return eval_list_A, eval_list_B
#
#     def ConsisVerif(self, img_tensor, threshold):
#         Verified_img_tensor = consistency.Binarization(img_tensor)
#         if threshold != 0:
#             Verified_img_tensor = consistency.RemoveSmallArea(
#                 img_tensor=Verified_img_tensor,
#                 threshold=threshold,
#                 median_ksize=3  # 中值滤波核大小，可改，奇数
#             )
#         return Verified_img_tensor
#
#     def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
#         os.makedirs('./Results' + savepath, exist_ok=True)
#
#         eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
#         eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
#         eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
#
#         cnt = 1
#         running_time = []
#         with torch.no_grad():
#             for A_tensor, B_tensor in eval_loader_tqdm:
#                 start_time = time.time()
#
#                 # ---------- 网络前向 ----------
#                 NetOut = model(A_tensor, B_tensor)  # shape: (1, 1, H, W)
#
#                 # ---------- 读取源图像（提前，用于导向滤波） ----------
#                 A_img = cv2.imread(eval_list_A[cnt - 1])
#                 B_img = cv2.imread(eval_list_B[cnt - 1])
#
#                 # ---------- 导向滤波细化概率图 ----------
#                 prob = NetOut[0, 0].detach().cpu().numpy().astype(np.float32)  # (H, W)
#
#                 # 生成引导图：两源图像的灰度平均（也可以只用一幅，根据任务调整）
#                 if A_img.ndim == 3:
#                     grayA = cv2.cvtColor(A_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#                     grayB = cv2.cvtColor(B_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#                 else:
#                     grayA = A_img.astype(np.float32) / 255.0
#                     grayB = B_img.astype(np.float32) / 255.0
#                 guide = (grayA + grayB) * 0.5
#
#                 # 导向滤波
#                 prob_refined = consistency.guided_filter(guide, prob,
#                                                          radius=self.guided_radius,
#                                                          eps=self.guided_eps)
#                 # 转回 Tensor
#                 prob_refined_tensor = torch.from_numpy(prob_refined).unsqueeze(0).unsqueeze(0).to(self.DEVICE)
#
#                 # ---------- 一致性检验（二值化+小区域移除+中值滤波） ----------
#                 D = self.ConsisVerif(prob_refined_tensor, threshold)[0]
#
#                 # ---------- 图像融合 ----------
#                 D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()  # (H, W, 1)
#                 Final_fused = A_img * D + B_img * (1 - D)
#
#                 # 保存结果
#                 cv2.imwrite('./Results' + savepath + '/lytro-' + str(cnt).zfill(2) + '-dm.png', D * 255)
#                 cv2.imwrite('./Results' + savepath + '/lytro-' + str(cnt).zfill(2) + '.png', Final_fused)
#
#                 running_time.append(time.time() - start_time)
#                 cnt += 1
#
#         for i, t in enumerate(running_time):
#             print("process_time: {} s".format(t))
#         print("\nResults are saved in: " + "./Results" + savepath)
#
#
# if __name__ == '__main__':
#     f = Fusion()
#     f()


# import os
# import sys
# import glob
# import time
#
# import cv2
# import numpy as np
# import torch
#
# from tqdm import tqdm
# from torch.utils.data import DataLoader
#
# import dataLoader as DLr
# from utils import consistency
# from network import MyNetwork
# from utils.cuda_chk import GPUorCPU
#
# class Fusion:
#     def __init__(
#             self,
#             modelpath='./pre-trained weights/best_network.pth',
#             dataroot='./Datasets/Eval',
#             dataset_name='Lytro',
#
#             # 网络Sigmoid概率图初始二值化阈值
#             initial_threshold=0.50,
#
#             # 【核心修改】：从 0.80 降低至 0.60，拯救弱焦点边缘
#             final_threshold=0.60,
#
#             # 曲率感知Domain Transform参数
#             sigma_s=10.0,
#             sigma_r=0.15,
#             rf_iterations=3,
#
#             # 是否保存阈值对照结果 (保留你的对照实验功能)
#             save_threshold_candidates=True
#     ):
#         self.DEVICE = torch.device(GPUorCPU().DEVICE)
#         self.MODELPATH = modelpath
#         self.DATAROOT = dataroot
#         self.DATASET_NAME = dataset_name
#         self.INITIAL_THRESHOLD = float(initial_threshold)
#         self.FINAL_THRESHOLD = float(final_threshold)
#         self.SIGMA_S = float(sigma_s)
#         self.SIGMA_R = float(sigma_r)
#         self.RF_ITERATIONS = int(rf_iterations)
#         self.SAVE_THRESHOLD_CANDIDATES = bool(save_threshold_candidates)
#
#         # 主融合结果
#         self.SAVEPATH = os.path.join('./Results', self.DATASET_NAME)
#         # 中间结果单独保存
#         self.DEBUGPATH = os.path.join('./Results', self.DATASET_NAME + '_RF_debug')
#         # 阈值对照实验
#         self.CANDIDATEPATH = os.path.join('./Results', self.DATASET_NAME + '_RF_threshold_candidates')
#         self.CANDIDATE_THRESHOLDS = [0.50, 0.60, 0.80]
#
#     def __call__(self, *args, **kwargs):
#         model = self.LoadWeights(self.MODELPATH)
#         eval_list_A, eval_list_B = self.PrepareData(os.path.join(self.DATAROOT, self.DATASET_NAME))
#
#         if len(eval_list_A) == 0:
#             raise FileNotFoundError('No sourceA images found!')
#         if len(eval_list_A) != len(eval_list_B):
#             raise ValueError('sourceA and sourceB image counts do not match!')
#
#         self.FusionProcess(model, eval_list_A, eval_list_B, self.SAVEPATH)
#
#     def LoadWeights(self, modelpath):
#         if not os.path.isfile(modelpath):
#             raise FileNotFoundError('Model weights not found: ' + modelpath)
#
#         model = MyNetwork().to(self.DEVICE)
#         try:
#             checkpoint = torch.load(modelpath, map_location=self.DEVICE, weights_only=False)
#         except TypeError:
#             checkpoint = torch.load(modelpath, map_location=self.DEVICE)
#
#         if isinstance(checkpoint, dict):
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             elif 'model_state_dict' in checkpoint:
#                 state_dict = checkpoint['model_state_dict']
#             elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
#                 state_dict = checkpoint['model']
#             else:
#                 state_dict = checkpoint
#         else:
#             state_dict = checkpoint
#
#         if isinstance(state_dict, dict):
#             state_dict = {
#                 (key[7:] if key.startswith('module.') else key): value
#                 for key, value in state_dict.items()
#             }
#
#         model.load_state_dict(state_dict, strict=True)
#         model.eval()
#
#         num_params = sum(p.numel() for p in model.parameters())
#         print('The number of model parameters: {:.6f} M'.format(num_params / 1e6))
#         print('Device:', self.DEVICE)
#         return model
#
#     def PrepareData(self, datapath):
#         eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
#         eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
#         return eval_list_A, eval_list_B
#
#     @staticmethod
#     def ExtractProbability(net_output):
#         if isinstance(net_output, (tuple, list)): net_output = net_output[0]
#         if not torch.is_tensor(net_output): raise TypeError('Network output must be a torch.Tensor')
#
#         if net_output.ndim == 4:
#             if net_output.shape[0] != 1 or net_output.shape[1] != 1:
#                 raise ValueError('Expected [1,1,H,W], ' f'but got {tuple(net_output.shape)}')
#             probability = net_output[0, 0]
#         elif net_output.ndim == 3:
#             if net_output.shape[0] != 1: raise ValueError('Expected one-channel output')
#             probability = net_output[0]
#         elif net_output.ndim == 2:
#             probability = net_output
#         else:
#             raise ValueError(f'Unsupported network output shape: {tuple(net_output.shape)}')
#
#         probability = probability.detach().float().cpu().numpy().astype(np.float32)
#         probability = np.nan_to_num(probability, nan=0.5, posinf=1.0, neginf=0.0)
#
#         if probability.min() < -1e-3 or probability.max() > 1.0 + 1e-3:
#             raise ValueError('Network output is outside [0,1]. Please check the Sigmoid output.')
#
#         return np.clip(probability, 0.0, 1.0)
#
#     def ConsisVerif(self, probability_tensor, image_A, image_B):
#         # 调用重构后的 consistency 模块
#         final_decision, filtered_map, guide = consistency.RefineDecisionMap(
#             prob_tensor=probability_tensor,
#             imgA=image_A,
#             imgB=image_B,
#             initial_threshold=self.INITIAL_THRESHOLD,
#             final_threshold=self.FINAL_THRESHOLD,  # 传入 0.6
#             sigma_s=self.SIGMA_S,
#             sigma_r=self.SIGMA_R,
#             num_iterations=self.RF_ITERATIONS,
#             min_area_ratio=0.005,  # 核心：恢复旧版的 0.5% 阈值
#             return_soft=True
#         )
#         return final_decision, filtered_map, guide
#
#     def FusionProcess(self, model, eval_list_A, eval_list_B, savepath):
#         os.makedirs(savepath, exist_ok=True)
#         os.makedirs(self.DEBUGPATH, exist_ok=True)
#
#         if self.SAVE_THRESHOLD_CANDIDATES:
#             for threshold in self.CANDIDATE_THRESHOLDS:
#                 candidate_dir = os.path.join(self.CANDIDATEPATH, 'th_{:.2f}'.format(threshold))
#                 os.makedirs(candidate_dir, exist_ok=True)
#
#         eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
#         eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False, num_workers=0)
#
#         progress = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout,
#                         desc='Fusion {}'.format(self.DATASET_NAME))
#         running_time = []
#
#         with torch.inference_mode():
#             for index, (A, B) in enumerate(progress, start=1):
#                 start_time = time.perf_counter()
#
#                 A = A.to(self.DEVICE, non_blocking=True)
#                 B = B.to(self.DEVICE, non_blocking=True)
#
#                 image_A = cv2.imread(eval_list_A[index - 1], cv2.IMREAD_COLOR)
#                 image_B = cv2.imread(eval_list_B[index - 1], cv2.IMREAD_COLOR)
#
#                 if image_A is None or image_B is None:
#                     raise FileNotFoundError('Source images not found.')
#
#                 if image_A.shape != image_B.shape:
#                     raise ValueError(f'Source image shape mismatch: {image_A.shape} vs {image_B.shape}')
#
#                 h, w = image_A.shape[:2]
#
#                 # 1. 网络预测
#                 net_output = model(A, B)
#                 probability_np = self.ExtractProbability(net_output)
#
#                 if probability_np.shape != (h, w):
#                     probability_np = cv2.resize(probability_np, (w, h), interpolation=cv2.INTER_LINEAR)
#
#                 probability_np = np.clip(probability_np, 0.0, 1.0).astype(np.float32)
#                 probability_tensor = torch.from_numpy(probability_np).unsqueeze(0).unsqueeze(0).to(self.DEVICE)
#
#                 # 2. 初始硬决策图 (用于对照)
#                 raw_decision = (probability_np >= self.INITIAL_THRESHOLD).astype(np.uint8)
#                 initial_fused = np.where(raw_decision[:, :, None] == 1, image_A, image_B).astype(np.uint8)
#
#                 # 3. 曲率感知 DT-RF 后处理 (包含桥接与去噪点)
#                 final_decision_tensor, filtered_tensor, guide = self.ConsisVerif(probability_tensor, image_A, image_B)
#
#                 filtered_map = filtered_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
#                 final_decision = (final_decision_tensor[0, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
#
#                 # 4. 最终原始硬选择融合
#                 final_fused = np.where(final_decision[:, :, None] == 1, image_A, image_B).astype(np.uint8)
#
#                 filename = os.path.basename(eval_list_A[index - 1])
#                 stem = os.path.splitext(filename)[0]
#
#                 # 5. 保存主结果
#                 cv2.imwrite(os.path.join(savepath, stem + '.png'), final_fused)
#                 cv2.imwrite(os.path.join(savepath, stem + '-dm.png'), final_decision * 255)
#
#                 # 6. 保存诊断数据
#                 np.save(os.path.join(self.DEBUGPATH, stem + '-prob.npy'), probability_np)
#                 cv2.imwrite(os.path.join(self.DEBUGPATH, stem + '-dm_raw.png'), raw_decision * 255)
#                 cv2.imwrite(os.path.join(self.DEBUGPATH, stem + '-dm_rf_soft.png'),
#                             np.clip(filtered_map * 255.0, 0, 255).astype(np.uint8))
#                 cv2.imwrite(os.path.join(self.DEBUGPATH, stem + '-guideY.png'),
#                             np.clip(guide * 255.0, 0, 255).astype(np.uint8))
#                 cv2.imwrite(os.path.join(self.DEBUGPATH, stem + '-initial_fused.png'), initial_fused)
#
#                 # 7. 阈值对照实验
#                 if self.SAVE_THRESHOLD_CANDIDATES:
#                     for threshold in self.CANDIDATE_THRESHOLDS:
#                         candidate_decision = (filtered_map >= threshold).astype(np.uint8)
#                         candidate_fused = np.where(candidate_decision[:, :, None] == 1, image_A, image_B).astype(
#                             np.uint8)
#                         candidate_dir = os.path.join(self.CANDIDATEPATH, 'th_{:.2f}'.format(threshold))
#
#                         cv2.imwrite(os.path.join(candidate_dir, stem + '.png'), candidate_fused)
#                         cv2.imwrite(os.path.join(candidate_dir, stem + '-dm.png'), candidate_decision * 255)
#
#                 running_time.append(time.perf_counter() - start_time)
#
#         # 8. 时间统计
#         for i, elapsed in enumerate(running_time, start=1):
#             print('Image {:02d} process_time: {:.4f} s'.format(i, elapsed))
#
#         if running_time:
#             print('\nAverage process_time: {:.4f} s'.format(np.mean(running_time)))
#
#         print('\nMain results saved in:', savepath)
#         print('Debug results saved in:', self.DEBUGPATH)
#         if self.SAVE_THRESHOLD_CANDIDATES:
#             print('Threshold candidates saved in:', self.CANDIDATEPATH)
#
#
# if __name__ == '__main__':
#     fusion = Fusion(
#         modelpath='./best_network_be.pth',
#         dataroot='./Datasets/Eval',
#         dataset_name='Lytro',
#         initial_threshold=0.50,
#         final_threshold=0.60,  # 【重要】已从 0.80 调至 0.60
#         sigma_s=10.0,
#         sigma_r=0.15,
#         rf_iterations=3,
#         save_threshold_candidates=True
#     )
#     fusion()


# import os
# import sys
# import glob
# import time
# import re
# import cv2
# import numpy as np
# import torch
#
# from tqdm import tqdm
# from torch import einsum
# import dataLoader as DLr
# from utils import consistency
# from network import MyNetwork
# from utils.cuda_chk import GPUorCPU
# from torch.utils.data import DataLoader
#
#
# class Fusion:
#     def __init__(self,
#                  modelpath='./best_network_be.pth',
#                  dataroot='./Datasets/Eval',
#                  dataset_name='LMIF',
#                  threshold=0.0015,
#                  save_root='./Results',
#                  seed_name=None,          # 新增：用于区分子目录，如 'seed_42'
#                  ):
#         self.DEVICE = GPUorCPU().DEVICE
#         self.MODELPATH = modelpath
#         self.DATAROOT = dataroot
#         self.DATASET_NAME = dataset_name
#         self.THRESHOLD = threshold
#         self.SAVE_ROOT = save_root
#         self.SEED_NAME = seed_name
#         self.DATAPATH = os.path.join(self.DATAROOT, self.DATASET_NAME)
#
#         # 输出子目录： ./Results/<数据集>/<seed>
#         if self.SEED_NAME:
#             self.SAVEPATH = f'/{self.DATASET_NAME}/{self.SEED_NAME}'
#         else:
#             self.SAVEPATH = f'/{self.DATASET_NAME}'
#
#     def __call__(self, *args, **kwargs):
#         if self.DATASET_NAME in ('Lytro', 'MFFW', 'MFI', 'Grayscale',
#                                  'Savic', 'HBU-CVMDSP', 'HRMFF', 'LowLight', 'LMIF'):
#             MODEL = self.LoadWeights(self.MODELPATH)
#             EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
#             self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
#         else:
#             print("Test Dataset required!")
#
#     def LoadWeights(self, modelpath):
#         model = MyNetwork().to(self.DEVICE)
#         model.load_state_dict(torch.load(modelpath, map_location=torch.device(self.DEVICE), weights_only=True))
#         model.eval()
#         num_params = sum(p.numel() for p in model.parameters())
#         print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
#         return model
#
#     def PrepareData(self, datapath):
#         eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
#         eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
#         return eval_list_A, eval_list_B
#
#     def ConsisVerif(self, img_tensor, threshold):
#         Verified_img_tensor = consistency.Binarization(img_tensor)
#         if threshold != 0:
#             Verified_img_tensor = consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
#         return Verified_img_tensor
#
#     def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
#         out_dir = self.SAVE_ROOT + savepath
#         os.makedirs(out_dir, exist_ok=True)
#
#         eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
#         eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
#         eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout,
#                                 desc=self.SEED_NAME or self.DATASET_NAME)
#
#         cnt = 1
#         running_time = []
#         with torch.no_grad():
#             for A, B in eval_loader_tqdm:
#                 start_time = time.time()
#                 NetOut = model(A, B)
#                 D = self.ConsisVerif(NetOut, threshold)[0]
#
#                 name_A = os.path.basename(eval_list_A[cnt - 1])
#                 name_B = os.path.basename(eval_list_B[cnt - 1])
#                 print(f"Pair {cnt}: A = {name_A}, B = {name_B}")
#
#                 D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()
#                 A_img = cv2.imread(eval_list_A[cnt - 1])
#                 B_img = cv2.imread(eval_list_B[cnt - 1])
#                 Final_fused = A_img * D + B_img * (1 - D)
#
#                 src_name = os.path.splitext(name_A)[0]
#                 match = re.search(r'\d+', src_name)
#                 num_str = match.group() if match else src_name
#
#                 cv2.imwrite(os.path.join(out_dir, f'lytro-{num_str}-dm.png'), D * 255)
#                 cv2.imwrite(os.path.join(out_dir, f'lytro-{num_str}.png'), Final_fused)
#
#                 running_time.append(time.time() - start_time)
#                 cnt += 1
#
#         avg_t = sum(running_time) / max(len(running_time), 1)
#         print(f"\n[{self.SEED_NAME}] pairs={len(running_time)} avg_time={avg_t:.4f}s")
#         print(f"[{self.SEED_NAME}] Results saved in: {out_dir}\n")
#
#
# def run_all_seeds(seed_root='./seed',
#                   pattern='*/best_network.pth',
#                   dataroot='./Datasets/Eval',
#                   dataset_name='LMIF',
#                   threshold=0.0015,
#                   save_root='./Results'):
#     weight_paths = sorted(glob.glob(os.path.join(seed_root, pattern)))
#     if not weight_paths:
#         print(f"No weight file found under {seed_root} with pattern {pattern}")
#         return
#
#     print(f"Found {len(weight_paths)} weight files:")
#     for p in weight_paths:
#         print("  ", p)
#
#     for wp in weight_paths:
#         seed_name = os.path.basename(os.path.dirname(wp))  # e.g. '42'
#         sub_name = seed_name if seed_name.startswith('seed') else f'seed_{seed_name}'
#
#         print("\n" + "=" * 72)
#         print(f"Running seed '{seed_name}' -> {save_root}/{dataset_name}/{sub_name}")
#         print("=" * 72)
#
#         try:
#             f = Fusion(modelpath=wp,
#                        dataroot=dataroot,
#                        dataset_name=dataset_name,
#                        threshold=threshold,
#                        save_root=save_root,
#                        seed_name=sub_name)
#             f()
#         finally:
#             # 及时释放模型，避免多个 seed 叠加导致显存爆掉
#             try:
#                 del f
#             except NameError:
#                 pass
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#     print("\nAll seeds finished.")
#
#
# if __name__ == '__main__':
#     run_all_seeds(
#         seed_root='./seed',
#         pattern='*/best_network.pth',
#         dataroot='./Datasets/Eval',
#         dataset_name='Lytro',
#         threshold=0.005,
#         save_root='./Results',
#     )


#seed
# import os
# import sys
# import glob
# import time
# import re
# import cv2
# import numpy as np
# import torch
#
# from tqdm import tqdm
# from torch import einsum
# import dataLoader as DLr
# from utils import consistency
# from network import MyNetwork
# from utils.cuda_chk import GPUorCPU
# from torch.utils.data import DataLoader
#
# # 支持的数据集名称（用于 __call__ 判断）
# SUPPORTED_DATASETS = ['Lytro', 'MFFW', 'MFI', 'Grayscale',
#                       'Savic', 'HBU-CVMDSP', 'HRMFF', 'LowLight', 'LMIF', 'Road-MF']
#
# # ============ 需要跑的权重文件夹（seed）============
# RUN_SEEDS = ['1', '2']
#
# # 每个 seed 都要跑的数据集
# # RUN_DATASETS = ['Lytro', 'MFFW', 'MFI', 'LMIF', 'Road-MF']
# RUN_DATASETS = ['Road-MF']
#
# # 权重文件模板：./seed/<seed>/best_network.pth
# DEFAULT_WEIGHT_TEMPLATE = './seed/{seed}/best_network.pth'
#
#
# class Fusion:
#     def __init__(self,
#                  modelpath='./seed/3/best_network.pth',  # 训练好的权重位置
#                  dataroot='./Datasets/Eval',             # 测试数据集根目录
#                  dataset_name='Road-MF',                 # 数据集名称
#                  threshold=0.0015,                       # 小区域移除阈值
#                  model=None,                             # 可选：外部传入已加载的模型
#                  seed=None,                              # 新增：对应的 seed 名（用于保存路径分层）
#                  ):
#         self.DEVICE = GPUorCPU().DEVICE
#         self.MODELPATH = modelpath
#         self.DATAROOT = dataroot
#         self.DATASET_NAME = dataset_name
#         self.THRESHOLD = threshold
#         self.SEED = seed
#         self.DATAPATH = os.path.join(self.DATAROOT, self.DATASET_NAME)
#
#         # 保存路径分层：./Results/seed4/Lytro 或 ./Results/Lytro
#         if self.SEED:
#             self.SAVEPATH = '/seed{}/{}'.format(self.SEED, self.DATASET_NAME)
#         else:
#             self.SAVEPATH = '/' + self.DATASET_NAME
#
#         self.model = model  # 若为 None，则在 __call__ 中加载
#
#     def __call__(self, *args, **kwargs):
#         if self.DATASET_NAME in SUPPORTED_DATASETS:
#             if self.model is None:
#                 self.model = self.LoadWeights(self.MODELPATH)
#             EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
#             if len(EVAL_LIST_A) == 0 or len(EVAL_LIST_B) == 0:
#                 print(f"[Warning] Dataset '{self.DATASET_NAME}' has no images, skip.")
#                 return
#             self.FusionProcess(self.model, EVAL_LIST_A, EVAL_LIST_B,
#                                self.SAVEPATH, self.THRESHOLD)
#         else:
#             print(f"Test Dataset '{self.DATASET_NAME}' not supported! "
#                   f"Supported: {SUPPORTED_DATASETS}")
#
#     def LoadWeights(self, modelpath):
#         model = MyNetwork().to(self.DEVICE)
#         state_dict = torch.load(modelpath,
#                                 map_location=torch.device(self.DEVICE),
#                                 weights_only=True)
#         model.load_state_dict(state_dict)
#         model.eval()
#
#         num_params = 0
#         for p in model.parameters():
#             num_params += p.numel()
#         print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
#         return model
#
#     def PrepareData(self, datapath):
#         eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
#         eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
#         return eval_list_A, eval_list_B
#
#     def ConsisVerif(self, img_tensor, threshold):
#         Verified_img_tensor = consistency.Binarization(img_tensor)
#         if threshold != 0:
#             Verified_img_tensor = consistency.RemoveSmallArea(
#                 img_tensor=Verified_img_tensor, threshold=threshold)
#         return Verified_img_tensor
#
#     def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
#         save_dir = './Results' + savepath
#         os.makedirs(save_dir, exist_ok=True)
#
#         eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
#         eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
#         eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
#
#         cnt = 1
#         running_time = []
#         with torch.no_grad():
#             for A, B in eval_loader_tqdm:
#                 start_time = time.time()
#                 NetOut = model(A, B)
#                 D = self.ConsisVerif(NetOut, threshold)[0]
#
#                 name_A = os.path.basename(eval_list_A[cnt - 1])
#                 name_B = os.path.basename(eval_list_B[cnt - 1])
#                 print(f"Pair {cnt}: A = {name_A}, B = {name_B}")
#
#                 D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()
#                 A_img = cv2.imread(eval_list_A[cnt - 1])
#                 B_img = cv2.imread(eval_list_B[cnt - 1])
#                 Final_fused = A_img * D + B_img * (1 - D)
#
#                 src_name = os.path.splitext(name_A)[0]
#                 match = re.search(r'\d+', src_name)
#                 num_str = match.group() if match else src_name
#
#                 # 文件名加 seed 前缀，避免不同 seed 混到一起时文件名冲突
#                 prefix = f'seed{self.SEED}-' if self.SEED else ''
#                 cv2.imwrite(os.path.join(save_dir, prefix + num_str + '-dm.png'), D * 255)
#                 cv2.imwrite(os.path.join(save_dir, prefix + num_str + '.png'), Final_fused)
#
#                 running_time.append(time.time() - start_time)
#                 cnt += 1
#
#         for t in running_time:
#             print("process_time: {} s".format(t))
#         print("\nResults are saved in: " + save_dir)
#
#
# if __name__ == '__main__':
#     DATAROOT = './Datasets/Eval'
#
#     # ===== 可选：为 (seed, dataset) 或 dataset 单独指定权重路径 =====
#     # 键可以是 (seed, dataset) 元组，也可以是 dataset 字符串
#     # 未匹配时回退到 DEFAULT_WEIGHT_TEMPLATE
#     MODELPATHS = {
#         # ('4', 'Lytro'): './seed/4/lytro_best_network.pth',
#         # 'MFI': './seed/common/mfi_best_network.pth',
#     }
#
#     total_start = time.time()
#
#     # 外层：遍历不同的权重文件夹（seed）
#     for seed in RUN_SEEDS:
#         print("\n" + "#" * 70)
#         print(f"### Running with weights of seed: {seed}")
#         print("#" * 70)
#
#         # 同一 seed 下，模型在多个数据集间复用
#         shared_model = None
#         shared_model_path = None
#
#         # 内层：遍历数据集
#         for name in RUN_DATASETS:
#             datapath = os.path.join(DATAROOT, name)
#
#             if not os.path.isdir(datapath):
#                 print(f"[Skip] Dataset not found: {datapath}")
#                 continue
#
#             if not (os.path.isdir(os.path.join(datapath, 'sourceA')) and
#                     os.path.isdir(os.path.join(datapath, 'sourceB'))):
#                 print(f"[Skip] Dataset '{name}' missing 'sourceA' or 'sourceB' folder.")
#                 continue
#
#             # 决定该 (seed, dataset) 使用哪个权重
#             model_path = MODELPATHS.get(
#                 (seed, name),
#                 MODELPATHS.get(name, DEFAULT_WEIGHT_TEMPLATE.format(seed=seed))
#             )
#
#             if not os.path.isfile(model_path):
#                 print(f"[Skip] Weight file not found: {model_path}")
#                 continue
#
#             print("\n" + "=" * 70)
#             print(f"Processing dataset: {name}  (seed={seed})")
#             print(f"Weight: {model_path}")
#             print("=" * 70)
#
#             # 若路径与上一次相同，直接复用模型；否则重新加载
#             model_to_use = shared_model if (shared_model is not None
#                                             and shared_model_path == model_path) else None
#
#             fusion = Fusion(modelpath=model_path,
#                             dataroot=DATAROOT,
#                             dataset_name=name,
#                             model=model_to_use,
#                             seed=seed)
#             fusion()
#
#             shared_model = fusion.model
#             shared_model_path = model_path
#
#     print("\n" + "=" * 70)
#     print("All seeds & datasets finished. Total time: {:.2f} s".format(time.time() - total_start))
#     print("=" * 70)


#data_test
# import os
# import sys
# import glob
# import time
# import re
# import cv2
# import numpy as np
# import torch
#
# from tqdm import tqdm
# from torch import einsum
# import dataLoader as DLr
# from utils import consistency
# from network import MyNetwork
# from utils.cuda_chk import GPUorCPU
# from torch.utils.data import DataLoader
#
# # 支持的数据集名称（用于 __call__ 判断）
# SUPPORTED_DATASETS = ['Lytro', 'MFFW', 'MFI', 'Grayscale',
#                       'Savic', 'HBU-CVMDSP', 'HRMFF', 'LowLight', 'LMIF', 'Road-MF']
#
# # ============ 待测权重根目录：每个子文件夹放一份权重 ============
# WEIGHTS_ROOT = './data_test'
#
# # 权重文件扩展名模式（按优先级搜索）
# WEIGHT_PATTERNS = ['*.pth', '*.pt', '*.pth.tar']
#
# # ============ 数据集 & 数据根目录 & 结果根目录 ============
# DATASET_NAME = 'Lytro'
# DATAROOT     = './Datasets/Eval'
# RESULT_ROOT  = './Results/data_test'   # 最终保存：./Results/data_test/<子文件夹名>/Lytro/
#
#
# class Fusion:
#     def __init__(self,
#                  modelpath,                     # 训练好的权重路径
#                  dataroot='./Datasets/Eval',    # 测试数据集根目录
#                  dataset_name='Lytro',          # 数据集名称
#                  savepath=None,                 # 结果保存目录（完整路径）
#                  threshold=0.0015,              # 小区域移除阈值
#                  model=None,                    # 可选：外部传入已加载的模型
#                  ):
#         self.DEVICE       = GPUorCPU().DEVICE
#         self.MODELPATH    = modelpath
#         self.DATAROOT     = dataroot
#         self.DATASET_NAME = dataset_name
#         self.THRESHOLD    = threshold
#         self.DATAPATH     = os.path.join(self.DATAROOT, self.DATASET_NAME)
#         self.SAVEPATH     = savepath
#         self.model        = model
#
#     def __call__(self, *args, **kwargs):
#         if self.DATASET_NAME not in SUPPORTED_DATASETS:
#             print(f"Test Dataset '{self.DATASET_NAME}' not supported! "
#                   f"Supported: {SUPPORTED_DATASETS}")
#             return
#
#         if self.model is None:
#             self.model = self.LoadWeights(self.MODELPATH)
#
#         EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
#         if len(EVAL_LIST_A) == 0 or len(EVAL_LIST_B) == 0:
#             print(f"[Warning] Dataset '{self.DATASET_NAME}' has no images, skip.")
#             return
#
#         self.FusionProcess(self.model, EVAL_LIST_A, EVAL_LIST_B,
#                            self.SAVEPATH, self.THRESHOLD)
#
#     def LoadWeights(self, modelpath):
#         model = MyNetwork().to(self.DEVICE)
#         state_dict = torch.load(modelpath,
#                                 map_location=torch.device(self.DEVICE),
#                                 weights_only=True)
#         model.load_state_dict(state_dict)
#         model.eval()
#
#         num_params = 0
#         for p in model.parameters():
#             num_params += p.numel()
#         print("The number of model parameters: {} M\n\n"
#               .format(round(num_params / 10e5, 6)))
#         return model
#
#     def PrepareData(self, datapath):
#         eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
#         eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
#         return eval_list_A, eval_list_B
#
#     def ConsisVerif(self, img_tensor, threshold):
#         Verified_img_tensor = consistency.Binarization(img_tensor)
#         if threshold != 0:
#             Verified_img_tensor = consistency.RemoveSmallArea(
#                 img_tensor=Verified_img_tensor, threshold=threshold)
#         return Verified_img_tensor
#
#     def FusionProcess(self, model, eval_list_A, eval_list_B, save_dir, threshold):
#         os.makedirs(save_dir, exist_ok=True)
#
#         eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
#         eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
#         eval_loader_tqdm = tqdm(eval_loader, colour='blue',
#                                 leave=True, file=sys.stdout)
#
#         cnt = 1
#         running_time = []
#         with torch.no_grad():
#             for A, B in eval_loader_tqdm:
#                 start_time = time.time()
#
#                 NetOut = model(A, B)
#                 D = self.ConsisVerif(NetOut, threshold)[0]
#
#                 name_A = os.path.basename(eval_list_A[cnt - 1])
#                 name_B = os.path.basename(eval_list_B[cnt - 1])
#                 print(f"Pair {cnt}: A = {name_A}, B = {name_B}")
#
#                 # c w h -> w h c
#                 D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()
#                 A_img = cv2.imread(eval_list_A[cnt - 1])
#                 B_img = cv2.imread(eval_list_B[cnt - 1])
#                 Final_fused = A_img * D + B_img * (1 - D)
#
#                 src_name = os.path.splitext(name_A)[0]
#                 match = re.search(r'\d+', src_name)
#                 num_str = match.group() if match else src_name
#
#                 cv2.imwrite(os.path.join(save_dir, num_str + '-dm.png'),
#                             (D * 255).astype(np.uint8))
#                 cv2.imwrite(os.path.join(save_dir, num_str + '.png'),
#                             Final_fused.astype(np.uint8))
#
#                 running_time.append(time.time() - start_time)
#                 cnt += 1
#
#         for t in running_time:
#             print("process_time: {} s".format(t))
#         if running_time:
#             print("avg process_time: {:.4f} s".format(
#                 sum(running_time) / len(running_time)))
#         print("\nResults are saved in: " + save_dir)
#
#
# def find_weight_in_folder(folder_path):
#     """在文件夹里按优先级寻找第一个权重文件，找不到返回 None。"""
#     for pattern in WEIGHT_PATTERNS:
#         files = sorted(glob.glob(os.path.join(folder_path, pattern)))
#         if files:
#             return files[0]
#     return None
#
#
# if __name__ == '__main__':
#     if not os.path.isdir(WEIGHTS_ROOT):
#         print(f"[Error] WEIGHTS_ROOT not found: {WEIGHTS_ROOT}")
#         sys.exit(1)
#
#     # 数据目录基本检查
#     data_path = os.path.join(DATAROOT, DATASET_NAME)
#     if not os.path.isdir(data_path):
#         print(f"[Error] Dataset path not found: {data_path}")
#         sys.exit(1)
#     if not (os.path.isdir(os.path.join(data_path, 'sourceA')) and
#             os.path.isdir(os.path.join(data_path, 'sourceB'))):
#         print(f"[Error] Dataset '{DATASET_NAME}' missing 'sourceA' or 'sourceB'.")
#         sys.exit(1)
#
#     total_start = time.time()
#
#     # 遍历 data_test 下的 6 个子文件夹
#     folders = sorted([d for d in os.listdir(WEIGHTS_ROOT)
#                       if os.path.isdir(os.path.join(WEIGHTS_ROOT, d))])
#     if not folders:
#         print(f"[Error] No sub-folder in {WEIGHTS_ROOT}")
#         sys.exit(1)
#
#     for folder_name in folders:
#         folder_path = os.path.join(WEIGHTS_ROOT, folder_name)
#         weight_path = find_weight_in_folder(folder_path)
#
#         if weight_path is None:
#             print(f"[Skip] No weight file in {folder_path} "
#                   f"(patterns: {WEIGHT_PATTERNS})")
#             continue
#
#         # 结果保存路径： ./Results/data_test/<folder_name>/Lytro
#         save_dir = os.path.join(RESULT_ROOT, folder_name, DATASET_NAME)
#
#         print("\n" + "=" * 70)
#         print(f"Processing folder: {folder_name}")
#         print(f"  Weight   : {weight_path}")
#         print(f"  Dataset  : {DATASET_NAME}  ({data_path})")
#         print(f"  Save to  : {save_dir}")
#         print("=" * 70)
#
#         fusion = Fusion(modelpath=weight_path,
#                         dataroot=DATAROOT,
#                         dataset_name=DATASET_NAME,
#                         savepath=save_dir)
#         fusion()
#
#     print("\n" + "=" * 70)
#     print("All folders finished. Total time: {:.2f} s"
#           .format(time.time() - total_start))
#     print("=" * 70)



import os
import sys
import glob
import time
import re
import cv2
import numpy as np
import torch

from tqdm import tqdm
from torch import einsum
import dataLoader as DLr
from utils import consistency
from network import MyNetwork
from utils.cuda_chk import GPUorCPU
from torch.utils.data import DataLoader

SUPPORTED_DATASETS = ['Lytro', 'MFFW', 'MFI', 'Grayscale',
                      'Savic', 'HBU-CVMDSP', 'HRMFF', 'LowLight', 'LMIF', 'Road-MF']

# ============ 待测权重根目录 ============
WEIGHTS_ROOT = './sequence_test'

# ============ 要跳过的文件夹（权重坏了 / 不跑） ============
SKIP_FOLDERS = ['M10']

WEIGHT_PATTERNS = ['*.pth', '*.pt', '*.pth.tar']

DATASET_NAME = 'Lytro'
DATAROOT     = './Datasets/Eval'
RESULT_ROOT  = './Results/sequence_test'

DEFAULT_N_SEQUENCE = 8
DEFAULT_ARW_RATIO  = 0.7


def parse_params_from_folder(folder_name):
    name = folder_name.strip()
    m = re.fullmatch(r'[Mm](\d+)', name)
    if m:
        return int(m.group(1)), DEFAULT_ARW_RATIO
    m = re.fullmatch(r'[Rr][Hh][Oo](\d+)', name)
    if m:
        return DEFAULT_N_SEQUENCE, int(m.group(1)) / 10.0
    return None, None


class Fusion:
    def __init__(self,
                 modelpath,
                 dataroot='./Datasets/Eval',
                 dataset_name='Lytro',
                 savepath=None,
                 threshold=0.0015,
                 model=None,
                 n_sequence=DEFAULT_N_SEQUENCE,
                 arw_length_ratio=DEFAULT_ARW_RATIO,
                 ):
        self.DEVICE           = GPUorCPU().DEVICE
        self.MODELPATH        = modelpath
        self.DATAROOT         = dataroot
        self.DATASET_NAME     = dataset_name
        self.THRESHOLD        = threshold
        self.DATAPATH         = os.path.join(self.DATAROOT, self.DATASET_NAME)
        self.SAVEPATH         = savepath
        self.model            = model
        self.N_SEQUENCE       = n_sequence
        self.ARW_LENGTH_RATIO = arw_length_ratio

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME not in SUPPORTED_DATASETS:
            print(f"Test Dataset '{self.DATASET_NAME}' not supported!")
            return

        if self.model is None:
            self.model = self.LoadWeights(self.MODELPATH)

        EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
        if len(EVAL_LIST_A) == 0 or len(EVAL_LIST_B) == 0:
            print(f"[Warning] Dataset '{self.DATASET_NAME}' has no images, skip.")
            return

        self.FusionProcess(self.model, EVAL_LIST_A, EVAL_LIST_B,
                           self.SAVEPATH, self.THRESHOLD)

    def LoadWeights(self, modelpath):
        model = MyNetwork(
            n_sequence=self.N_SEQUENCE,
            arw_length_ratio=self.ARW_LENGTH_RATIO,
        ).to(self.DEVICE)

        state_dict = torch.load(modelpath,
                                map_location=torch.device(self.DEVICE),
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        print("The number of model parameters: {} M\n\n"
              .format(round(num_params / 10e5, 6)))
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = consistency.Binarization(img_tensor)
        if threshold != 0:
            Verified_img_tensor = consistency.RemoveSmallArea(
                img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, save_dir, threshold):
        os.makedirs(save_dir, exist_ok=True)

        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
        eval_loader_tqdm = tqdm(eval_loader, colour='blue',
                                leave=True, file=sys.stdout)

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

                cv2.imwrite(os.path.join(save_dir, num_str + '-dm.png'),
                            (D * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_dir, num_str + '.png'),
                            Final_fused.astype(np.uint8))

                running_time.append(time.time() - start_time)
                cnt += 1

        for t in running_time:
            print("process_time: {} s".format(t))
        if running_time:
            print("avg process_time: {:.4f} s".format(
                sum(running_time) / len(running_time)))
        print("\nResults are saved in: " + save_dir)


def find_weight_in_folder(folder_path):
    for pattern in WEIGHT_PATTERNS:
        files = sorted(glob.glob(os.path.join(folder_path, pattern)))
        if files:
            return files[0]
    return None


if __name__ == '__main__':
    if not os.path.isdir(WEIGHTS_ROOT):
        print(f"[Error] WEIGHTS_ROOT not found: {WEIGHTS_ROOT}")
        sys.exit(1)

    data_path = os.path.join(DATAROOT, DATASET_NAME)
    if not os.path.isdir(data_path):
        print(f"[Error] Dataset path not found: {data_path}")
        sys.exit(1)
    if not (os.path.isdir(os.path.join(data_path, 'sourceA')) and
            os.path.isdir(os.path.join(data_path, 'sourceB'))):
        print(f"[Error] Dataset '{DATASET_NAME}' missing 'sourceA' or 'sourceB'.")
        sys.exit(1)

    total_start = time.time()

    # 按固定顺序跑
    preferred_order = ['M4', 'M6', 'M10', 'rho03', 'rho05', 'rho09']
    existing = {d for d in os.listdir(WEIGHTS_ROOT)
                if os.path.isdir(os.path.join(WEIGHTS_ROOT, d))}
    folders = [f for f in preferred_order if f in existing] + \
              [f for f in sorted(existing) if f not in preferred_order]

    if not folders:
        print(f"[Error] No sub-folder in {WEIGHTS_ROOT}")
        sys.exit(1)

    for folder_name in folders:
        # ---------- 跳过黑名单 ----------
        if folder_name in SKIP_FOLDERS:
            print(f"[Skip] {folder_name} 在 SKIP_FOLDERS 中，直接跳过。")
            continue

        folder_path = os.path.join(WEIGHTS_ROOT, folder_name)
        weight_path = find_weight_in_folder(folder_path)

        if weight_path is None:
            print(f"[Skip] No weight file in {folder_path}")
            continue

        n_seq, arw_ratio = parse_params_from_folder(folder_name)
        if n_seq is None or arw_ratio is None:
            print(f"[Skip] Cannot parse hyper-params from '{folder_name}'")
            continue

        save_dir = os.path.join(RESULT_ROOT, folder_name, DATASET_NAME)

        print("\n" + "=" * 70)
        print(f"Processing folder: {folder_name}")
        print(f"  n_sequence       : {n_seq}")
        print(f"  arw_length_ratio : {arw_ratio}")
        print(f"  Weight           : {weight_path}")
        print(f"  Dataset          : {DATASET_NAME}  ({data_path})")
        print(f"  Save to          : {save_dir}")
        print("=" * 70)

        fusion = Fusion(modelpath=weight_path,
                        dataroot=DATAROOT,
                        dataset_name=DATASET_NAME,
                        savepath=save_dir,
                        n_sequence=n_seq,
                        arw_length_ratio=arw_ratio)
        fusion()

    print("\n" + "=" * 70)
    print("All folders finished. Total time: {:.2f} s"
          .format(time.time() - total_start))
    print("=" * 70)