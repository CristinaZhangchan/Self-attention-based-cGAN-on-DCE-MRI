# python3 train.py --dataroot prostate_dataset --name T2_ADC_T1_to_DCE1_DCE2 --gpu_ids 0 --model aad_dce --which_model_netG res_cnn --which_model_netD aad
# --which_direction AtoB --lambda_A 10 --dataset_mode aligned --norm batch --pool_size 0 --output_nc 2 --input_nc 3 --loadSize 160 --fineSize 160
# --niter 50 --niter_decay 50 --checkpoints_dir checkpoints/ --display_id 0 --lr 0.0002

# python3 test.py --dataroot prostate_dataset --name T2_ADC_T1_to_DCE1_DCE2 --gpu_ids 0 --model aad_dce --which_model_netG res_cnn 
# --dataset_mode aligned --norm batch --phase test --output_nc 2 --input_nc 3 --how_many 10000 --serial_batches --fineSize 160 --loadSize 160 
# --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest

../crisenv/bin/python3 train.py  --dataroot /home/maia-user/cris/datasets/HER2_train_breasthollow_new/png_slices  --name fujian_b1000_breasthollow_new  --model aad_dce  --dataset_mode breast  --input_nc 3  --output_nc 3  --loadSize 320   --fineSize 320  --mask_size 320  --batchSize 4  --niter 100  --niter_decay 100  --gpu_ids 0   --display_freq 100   --print_freq 50   --save_epoch_freq 5


../crisenv/bin/python3 test.py   --dataroot /home/maia-user/cris/datasets/HER2_test_breasthollow_new/png_slices  --name fujian_b1000_breasthollow_new  --gpu_ids 0   --model aad_dce  --dataset_mode breast   --input_nc 3  --output_nc 3  --loadSize 320   --fineSize 320  --mask_size 320  --phase test --how_many 100000000