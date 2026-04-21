import os.path
import torch
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
from scipy.ndimage import center_of_mass
import torchvision.transforms as transforms

class BreastDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase 
        
        # 1. 定义你的文件夹路径 (根据截图)(Fujian)
        self.dir_tirm= os.path.join(opt.dataroot, 'DWI_1000')
        self.dir_adc = os.path.join(opt.dataroot, 'ADC')
        self.dir_p0 = os.path.join(opt.dataroot, 'DCE_phase0')
        self.dir_others = os.path.join(opt.dataroot, 'DCE_phaseOthers')
        self.dir_mask = os.path.join(opt.dataroot, 'segmentation') 

        # 1. 定义你的文件夹路径 (Bmmr2)
        # self.dir_tirm= os.path.join(opt.dataroot, 'DWI_all')
        # self.dir_adc = os.path.join(opt.dataroot, 'ADC_all')
        # self.dir_p0 = os.path.join(opt.dataroot, 'DCE_phase0')
        # self.dir_others = os.path.join(opt.dataroot, 'DCE_phaseOthers')
        # self.dir_mask = os.path.join(opt.dataroot, 'segmentation') 

        # 2. 获取基准文件列表 (以 TIRM 为主键)
        self.tirm_paths = sorted(self.make_dataset(self.dir_tirm))
        self.dataset_size = len(self.tirm_paths)

        # 3. 定义尺寸
        self.img_size = 320      # 全局分辨率
        self.crop_size = 128     # 局部关注区域大小 (建议比原先的60大，因为分辨率变大了)

        # 基础变换: ToTensor + Normalize (-1 to 1)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.png'):
                    images.append(os.path.join(root, fname))
        return images

    def __getitem__(self, index):
        # ---------------------------------------------------------
        # 1. 路径解析与匹配
        # ---------------------------------------------------------
        tirm_path = self.tirm_paths[index]
        basename = os.path.basename(tirm_path)
        # 例子: 19042580_slice_001_TIRM.png -> ID: 19042580_slice_001
        # 我们需要切掉最后部分的 _TIRM.png 来匹配其他文件
        # file_id = basename.replace('_b800.png', '') # bmmr2
        file_id = basename.replace('_DWI_b1000.png', '') # fujian
        
        # 构造其他模态的文件名 (fujian)
        adc_path = os.path.join(self.dir_adc, f"{file_id}_ADC.png")
        p0_path = os.path.join(self.dir_p0, f"{file_id}_DCE_phase0.png")
        
        # Target: Phase 1, 2, 3
        p1_path = os.path.join(self.dir_others, f"{file_id}_DCE_phase1.png")
        p2_path = os.path.join(self.dir_others, f"{file_id}_DCE_phase2.png")
        p3_path = os.path.join(self.dir_others, f"{file_id}_DCE_phase3.png")

        mask_path = os.path.join(self.dir_mask, f"{file_id}_seg.png")

        # # 构造其他模态的文件名 (bmmr2)
        # adc_path = os.path.join(self.dir_adc, f"{file_id}_adc.png")
        # p0_path = os.path.join(self.dir_p0, f"{file_id}_phase0.png")
        
        # # Target: Phase 1, 2, 3
        # p1_path = os.path.join(self.dir_others, f"{file_id}_phase1.png")
        # p2_path = os.path.join(self.dir_others, f"{file_id}_phase2.png")
        # p3_path = os.path.join(self.dir_others, f"{file_id}_phase3.png")

        # mask_path = os.path.join(self.dir_mask, f"{file_id}_seg.png")

        # ---------------------------------------------------------
        # 2. 图片读取 (转为灰度 'L')
        # ---------------------------------------------------------
        def read_img(path, is_mask=False):
            if os.path.exists(path):
                img = Image.open(path).convert('L')
                img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
                if is_mask:
                    return np.array(img) # Mask 返回 numpy 用于计算坐标
                return self.transform(img) # 图片返回 Tensor
            else:
                
                # 如果真要返回纯黑，在 Normalize 空间里纯黑是 -1，不是 0
                if is_mask: 
                    return np.zeros((self.img_size, self.img_size))
                return torch.ones((1, self.img_size, self.img_size)) * -1.0

        # 读取 Input
        t_tirm = read_img(tirm_path)
        t_adc = read_img(adc_path)
        t_p0 = read_img(p0_path)
        
        # 读取 Target
        t_p1 = read_img(p1_path)
        t_p2 = read_img(p2_path)
        t_p3 = read_img(p3_path)

        # 读取 Mask
        mask_np = read_img(mask_path, is_mask=True)

        # ---------------------------------------------------------
        # 3. 堆叠通道
        # ---------------------------------------------------------
        # Input A (3 Channels): TIRM, ADC, Phase0
        img_A = torch.cat([t_tirm, t_adc, t_p0], 0) 
        
        # Output B (3 Channels): Phase 1, 2, 3
        img_B = torch.cat([t_p1, t_p2, t_p3], 0)

        # ---------------------------------------------------------
        # 4. 动态坐标计算 (核心修改)
        # ---------------------------------------------------------
        # 如果 Mask 有值，计算重心；否则使用图片中心
        if np.max(mask_np) > 0:
            coords = center_of_mass(mask_np) # 返回 (y, x)
            center_y, center_x = int(coords[0]), int(coords[1])
        else:
            center_y, center_x = self.img_size // 2, self.img_size // 2

        # 计算左上角坐标 (x1, y1)，确保裁剪框不超出图像边界
        half = self.crop_size // 2
        y1 = max(0, min(center_y - half, self.img_size - self.crop_size))
        x1 = max(0, min(center_x - half, self.img_size - self.crop_size))
        y2 = y1 + self.crop_size
        x2 = x1 + self.crop_size
        
        # 裁剪 Target 的局部块 (Local Ground Truth)
        # [Channels, Height, Width]
        img_B_local = img_B[:, y1:y2, x1:x2]

        return {
            'A': img_A, 
            'B': img_B, 
            'B_local': img_B_local,
            'crop_coords': torch.tensor([y1, y2, x1, x2]), # 传给模型
            'A_paths': tirm_path,
            'DX': index
        }

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'BreastDataset'