# DCE-MRI
This repository implements an enhanced Conditional GAN (cGAN) based on the pix2pix architecture, specifically optimized for medical image synthesis. The model incorporates a Self-Attention Mechanism to improve the spatial correlation between global breast anatomy and localized tumor regions.

## Key Innovations:

1. Targeted Attention: Leverages self-attention to "flexibly lock" the relationship between the breast mask and the tumor, ensuring high-fidelity synthesis of tumor morphology.

2. Dynamic Phase Modeling: Specifically designed to capture subtle tumor variations across different imaging phases.

3. Multi-Scale Consistency: Balances global structural integrity with local textural details, preventing distortion in the breast boundary while focusing on the lesion area.

### Training:
```bash
 python train.py \
  --dataroot datapath \
  --name  \
  --model aad_dce \
  --dataset_mode breast \
  --input_nc 3 \
  --output_nc 3 \
  --loadSize 320 \
  --fineSize 320 \
  --mask_size 320 \
  --batchSize 4 \
  --niter 100 \
  --niter_decay 100 \
  --gpu_ids 0 \
  --display_freq 100 \
  --print_freq 50 \
  --save_epoch_freq
```
### Testing
```bash
python test.py \
  --dataroot datapath
  --name  \
  --model aad_dce \
  --dataset_mode breast \
  --input_nc 3 \
  --output_nc 3 \
  --loadSize 320 \
  --fineSize 320 \
  --mask_size 320 \
  --phase test \
  --how_many 100000
```
# Self-attention-based-cGAN-on-DCE-MRI
# Self-attention-based-cGAN-on-DCE-MRI
# Self-attention-based-cGAN-on-DCE-MRI
