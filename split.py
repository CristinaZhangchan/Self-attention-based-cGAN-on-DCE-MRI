import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import util.util as util
from PIL import Image
import numpy as np

def save_split_phases(image_numpy, file_id, base_dir, type_name):
    """
    保存拆分后的 Phase 1, 2, 3
    image_numpy: [H, W, 3] 范围 0-255
    file_id: 例如 "19042580_slice_001"
    base_dir: 保存的根目录
    type_name: "real" 或 "fake"
    """
    # 1. 创建分开的文件夹: e.g., results/.../split/real/
    save_dir = os.path.join(base_dir, type_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 定义 Phase 名称
    phases = ['phase1', 'phase2', 'phase3']
    
    # 3. 循环拆分保存
    for channel_idx, phase_name in enumerate(phases):
        # 提取单通道 (H, W)
        single_channel = image_numpy[:, :, channel_idx].astype(np.uint8)
        
        # 构建文件名: patientid_slice_001_real_phase1.png
        filename = f"{file_id}_{type_name}_{phase_name}.png"
        filepath = os.path.join(save_dir, filename)
        
        # 保存
        Image.fromarray(single_channel).save(filepath)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    
    # 创建结果目录
    web_dir = '/home/maia-user/cris/AAD-DCE/results/breast_cancer_dce_v1/test_latest/images'
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    
    # 定义拆分图片的保存根目录
    split_dir = '/home/maia-user/cris/AAD-DCE/split_output/'
    split_dir = os.path.join(web_dir, 'split_output')
    
    print(f"Starting testing. Split images will be saved to: {split_dir}")

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        # 1. 运行模型
        model.set_input(data)
        model.test() # forward pass
        
        # 2. 获取原始文件 ID
        # data['A_paths'] 是列表，取第一个元素
        # 路径类似: .../19042580_slice_001_TIRM.png
        img_path = model.get_image_paths()[0]
        basename = os.path.basename(img_path)
        
        # 解析 ID: 移除后缀 (_TIRM.png)
        # 结果: 19042580_slice_001
        file_id = basename.replace('_TIRM.png', '').replace('_TIRM.jpg', '')
        
        # 3. 获取图像数据 (numpy array, 0-255, HxWx3)
        # visuals 包含 'real_A', 'fake_B', 'real_B'
        visuals = model.get_current_visuals()
        
        # 4. 保存常规 HTML 结果 (可选，保持原样以便快速浏览)
        print('%04d: process image... %s' % (i, file_id))
        visualizer.save_images(webpage, visuals, [img_path], aspect_ratio=opt.aspect_ratio)
        
        # 5. --- 核心修改: 拆分并保存 Real B 和 Fake B ---
        
        # 处理 Real B (Ground Truth)
        if 'real_B' in visuals:
            save_split_phases(visuals['real_B'], file_id, split_dir, 'real')
            
        # 处理 Fake B (Prediction)
        if 'fake_B' in visuals:
            save_split_phases(visuals['fake_B'], file_id, split_dir, 'fake')

    webpage.save()
    print("Testing finished.")