export CUDA_VISIBLE_DEVICES=9
python evaluate_depth.py \
    --data_path ./data/kitti_depth/input \
    --load_weights_folder /data1/dataset/szy_code/monovit/log/baseline/mpsmall_2gpu_4bs_ms/models/weights_19 \
    --eval_mono \
    --png \
    --batch_size 1 \
    --eval_split eigen \

