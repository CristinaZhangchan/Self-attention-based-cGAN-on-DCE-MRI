import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import torchvision.transforms as transforms
from data import aux_dataset
import itertools
import cv2 as cv
import numpy as np

class aad_dce_model(BaseModel):
    def name(self):
        return 'aad_dce_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.mask_size = opt.mask_size 
 
        # 定义生成器
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, 
                                      img_size=opt.loadSize, 
                                      norm=opt.norm, use_dropout=not opt.no_dropout, init_type=opt.init_type, gpu_ids=self.gpu_ids)

        # 定义空间组件
        self.netSC = networks.define_C(norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        # 初始化 Attention Dataset
        self.aux_data = aux_dataset.AuxAttnDataset(7000, 7000, self.gpu_ids[0], mask_size=self.mask_size)
        
        # 定义判别器
        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            # input_nc + output_nc = 6 (3+3)
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD, 
                                          img_size=opt.loadSize,
                                          n_layers_D=opt.n_layers_D, norm=opt.norm, use_sigmoid=use_sigmoid, 
                                          init_type=opt.init_type, gpu_ids=self.gpu_ids, init_gain=opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.MSE_Loss = torch.nn.MSELoss()
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B'].to(torch.float)
        input_B = input['B' if AtoB else 'A'].to(torch.float)
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
            
        self.input_A = input_A
        self.input_B = input_B
        
        # --- 修复: 获取 image_paths ---
        # 如果是 aligned/unaligned 模式，key 是 'A_paths'
        if 'A_paths' in input:
            self.image_paths = input['A_paths']
        else:
            self.image_paths = [] # 防止报错

        # 获取动态坐标 [B, 4]
        if 'crop_coords' in input:
            self.crop_coords = input['crop_coords']
        else:
            # Fallback
            h, w = input_A.shape[2], input_A.shape[3]
            cs = 128
            y1 = (h - cs) // 2
            x1 = (w - cs) // 2
            self.crop_coords = torch.tensor([[y1, y1+cs, x1, x1+cs]]).to(self.gpu_ids[0])

        if 'B_local' in input:
            self.t_crop_precalc = input['B_local'].to(torch.float).cuda(self.gpu_ids[0])
        else:
            self.t_crop_precalc = None

        self.attn_A_index = input['DX'] 
        self.attn_A, _= self.aux_data.get_attn_map(self.attn_A_index, 0)

    def forward(self):
        self.real_A = Variable(self.input_A)
        concat_attn_A = self.attn_A
        
        # 生成器生成 Fake B
        self.fake_B = self.netG(self.real_A * (1. + concat_attn_A))
        self.real_B = Variable(self.input_B)
        
        # --- 动态裁剪逻辑 (适配 320x320) ---
        crops_fake = []
        crops_real = []
        
        for b in range(self.fake_B.shape[0]):
            coords = self.crop_coords[b] if isinstance(self.crop_coords, list) else self.crop_coords[b]
            y1, y2, x1, x2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            
            # Crop Fake
            c_f = self.fake_B[b, :, y1:y2, x1:x2]
            crops_fake.append(c_f.unsqueeze(0))
            
            # Crop Real
            if self.t_crop_precalc is not None:
                c_r = self.t_crop_precalc[b].unsqueeze(0)
            else:
                c_r = self.real_B[b, :, y1:y2, x1:x2].unsqueeze(0)
            crops_real.append(c_r)
            
        self.o_crop = torch.cat(crops_fake, dim=0)
        self.t_crop = torch.cat(crops_real, dim=0)

    def test(self):
        # 修复: 直接调用 forward()，确保所有变量 (real_A, fake_B) 都被正确赋值
        with torch.no_grad():
            self.forward()

    # --- 修复: 添加 get_image_paths ---
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Global
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        
        # Local
        local_A_list = []
        for b in range(self.real_A.shape[0]):
            coords = self.crop_coords[b]
            y1, y2, x1, x2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            local_A_list.append(self.real_A[b, :, y1:y2, x1:x2].unsqueeze(0))
        local_A = torch.cat(local_A_list, dim=0)
        
        fake_AB_local = torch.cat((local_A, self.o_crop), 1)
        real_AB_local = torch.cat((local_A, self.t_crop), 1)

        # Fake Loss
        pred_fake, _ = self.netD(fake_AB_local.detach(), fake_AB.detach(), self.crop_coords)
        fake_label = torch.zeros(pred_fake.shape).cuda(self.gpu_ids[0]) 
        self.loss_D_fake = self.MSE_Loss(pred_fake, fake_label) 

        # Real Loss
        pred_real, _ = self.netD(real_AB_local, real_AB, self.crop_coords)
        real_label = torch.ones(pred_real.shape).cuda(self.gpu_ids[0]) 
        self.loss_D_real = self.MSE_Loss(pred_real, real_label) 
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_adv
        self.loss_D.backward()

    def backward_G(self):
        # Global
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        
        # Local
        local_A_list = []
        for b in range(self.real_A.shape[0]):
            coords = self.crop_coords[b]
            y1, y2, x1, x2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            local_A_list.append(self.real_A[b, :, y1:y2, x1:x2].unsqueeze(0))
        local_A = torch.cat(local_A_list, dim=0)
        fake_AB_local = torch.cat((local_A, self.o_crop), 1)

        # G Loss
        pred_fake, self.tmp_attn = self.netD(fake_AB_local, fake_AB, self.crop_coords)
        real_label = torch.ones(pred_fake.shape).cuda(self.gpu_ids[0]) 
        
        self.loss_G_GAN = self.MSE_Loss(pred_fake, real_label) * self.opt.lambda_adv
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step() 
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.aux_data.update_attn_map(self.attn_A_index, self.tmp_attn.detach(), True)

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        # 保护性检查
        if not hasattr(self, 'real_A'):
            return OrderedDict()
            
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)