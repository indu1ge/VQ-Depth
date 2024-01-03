
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch --nproc_per_node=1 -m train_depth_vqvae \
  configs/vqvae.py \
  --local_rank 0