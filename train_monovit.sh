DATA_PATH="./data/kitti_depth/input"
exp=$1
model_name=$2
GPU_NUM=$3
BS=$4
PY_ARGS=${@:5}

EXP_DIR=/data1/dataset/szy_code/monovit/log/$exp
LOG_DIR=$EXP_DIR/$model_name
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi


export CUDA_VISIBLE_DEVICES=8,9
python -m torch.distributed.launch --nproc_per_node $GPU_NUM --master_port 2542 -m train \
    --dataset kitti \
    --data_path $DATA_PATH \
    --log_dir $EXP_DIR  \
    --model_name $model_name \
    --split eigen_zhou \
    --height 320 \
    --width 1024 \
    --png \
    --batch_size $BS \
    --num_workers 12 \
    --learning_rate 5e-5 \
    --num_epochs 20 \
    $PY_ARGS | tee -a $EXP_DIR/$model_name/log_train.txt    