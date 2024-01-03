image_size = 480
model = dict(
    image_size=image_size,
    num_resnet_blocks=2,
    downsample_ratio=32,
    num_tokens=128,
    codebook_dim=512,
    hidden_dim=16,
    use_norm=False,
    channels=1,
    train_objective='regression',
    max_value=10.,
    residul_type='v1',
    loss_type='si_log',
)

train_setting = dict(
    output_dir='outputs/test/vq/CS_bs24_hr_nocodebook',
    data=dict(
        dataset_mode='train',
        data_path='data/cityscapes',
        split_file='splits/cityscapes/cityscapes_train.txt',
        mask=True,
        mask_ratio=0.5,
        mask_patch_size=16,
        # crop_size=(image_size, image_size),
    ),
    opt_params=dict(
        epochs=20,
        batch_size=32,
        learning_rate=6e-4,
        lr_decay_rate=0.98,
        schedule_step=500,
        schedule_type='exp',
    )
)

test_setting = dict(
    data=dict(
        dataset_mode='eval',
        data_path='data/cityscapes',
        split_file='splits/cityscapes/cityscapes_val.txt',
        mask=False,
    ),
)