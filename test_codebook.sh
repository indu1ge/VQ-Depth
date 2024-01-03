
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch --nproc_per_node=1 -m test_vqvae \
  configs/vqvae.py \
  --local_rank 0 \
  --eval /ssd/szy/VQDepth/outputs/vq/CS_bs24_hr_nocodebook/vae-final.pt \