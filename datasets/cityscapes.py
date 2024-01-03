import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as tf
import json
import torch.nn.functional as F

from datasets.utils.data_reader import get_input_img
from datasets.utils.my_transforms import BatchRandomCrop, NoneTransform
from path_my import Path

def MaskGenerator(depth, mask_ratio=0.10, mask_patch_size=16):
    B, H, W = depth.shape
    assert H % mask_patch_size == 0 and W % mask_patch_size == 0 and mask_ratio > 0.0

    rand_size = (H // mask_patch_size) * (W // mask_patch_size)
    mask_count = int(np.ceil(rand_size * mask_ratio))

    mask_idx = np.random.permutation(rand_size)[:mask_count]
    mask = np.zeros(rand_size, dtype=float)
    mask[mask_idx] = 1.0

    mask = 1 - mask.reshape(B, (H // mask_patch_size), (W // mask_patch_size))
    mask = torch.from_numpy(mask.repeat(
        mask_patch_size, axis=1).repeat(mask_patch_size, axis=2))
    mask_depth = mask * depth
    return mask_depth

class CityscapesColorDataset(data.Dataset):

    def __init__(self, 
                 dataset_mode,
                 data_path='data/cityscapes',
                 split_file='splits/cityscapes',
                 crop_coords=[150, 0, 768, 2048],
                 full_size=[320, 1024],
                 patch_size=None,
                 normalize_params=[0.411, 0.432, 0.45],
                 depth_scale=256,
                 flip_mode=None, # "img", "k", "both", "semantic"(lr, ud, rotation)
                 load_disp=False,
                 use_casser_test=True,
                 load_test_gt=True,
                 mask=False,
                 mask_ratio=0.10,
                 mask_patch_size=16):
        self.init_opts = locals()

        self.dataset_mode = dataset_mode
        self.dataset_dir = data_path
        self.split_file = split_file

        self.crop_coords = crop_coords
        self.full_size = full_size
        self.patch_size = patch_size
        self.flip_mode = flip_mode

        self.load_disp = load_disp

        self.use_casser_test = use_casser_test
        self.load_test_gt = load_test_gt

        self.depth_scale = depth_scale

        self.file_list = self._get_file_list(split_file)

        self.mask = mask
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        

        if self.load_disp:
            assert self.flip_mode != "semantic",\
                "Images can't br rotated and flipped up and down when reading matching information."
        if self.load_test_gt:
            assert not self.load_disp,\
                "When the ground-truth depths are used, do not load the raw dispairties."
        
        # Initializate transforms
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=normalize_params, std=[1, 1, 1])
        if dataset_mode == "train":
            # random resize and crop
            if self.patch_size is not None:
                self.crop = BatchRandomCrop(patch_size)
            else:
                self.crop = NoneTransform()
            # just resize
            if self.full_size is not None and self.patch_size is None:
                pass
        else:
            # if self.load_test_gt:
            #     self.gt_depths = os.path.join(self.dataset_dir, "gt_depths")
            # if self.use_casser_test:
            #     self.crop_coords = [0, 0, 768, 2048]
            #     self.full_size = [192, 512]
            # if self.full_size is not None:
            #     self.color_resize = tf.Resize(self.full_size,
            #                                   interpolation=Image.ANTIALIAS)
            # else:
            #     self.color_resize = NoneTransform()
            pass

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, f_idx):
        '''Reture a data dictionary'''
        file_info = self.file_list[f_idx]
       
        # Read RGB-L -> inputs['color_s_raw'] original resolution
        inputs = {}
        color_l_path = file_info.split(' ')[0]
        color_l_path = os.path.join(self.dataset_dir, 'leftImg8bit', color_l_path)
        inputs['color_s_raw'] = get_input_img(color_l_path)
        # img_W = self.crop_coords[3] - self.crop_coords[1]
        # img_H = self.crop_coords[2] - self.crop_coords[0]

        # Read Depth -> inputs['depth'] orignal resolution
        disp_path = file_info.split(' ')[1]
        disp_path = os.path.join(self.dataset_dir, 'disparity', disp_path)
        
        c_params_path = file_info.split(" ")[2]
        c_params_path = os.path.join(self.dataset_dir, 'camera', c_params_path)
        with open(c_params_path) as f:
            camera = json.load(f)
        baseline        = camera['extrinsic']['baseline']
        focal_length    = camera['intrinsic']['fx']
        
        inputs['disparity'] = Image.open(disp_path)


        inputs['direct'] = torch.tensor(1, dtype=torch.float)

        # Process data
        # resize crop & color jit & flip(roataion) for train
        if self.dataset_mode == "train":
            # crop for image
            if self.crop_coords is not None:
                self.fix_crop = tf.functional.crop
            else:
                self.fix_crop = NoneTransform()

            # flip
            is_flip = (self.dataset_mode == 'train' and
                       random.uniform(0, 1) > 0.5)
            
            # resize
            if self.full_size is not None:
                img_size = self.full_size

                self.color_resize = tf.Resize(img_size, interpolation=Image.ANTIALIAS)
                self.map_resize = tf.Resize(img_size, interpolation=Image.NEAREST)
            else:
                self.color_resize = NoneTransform()
                self.map_resize = NoneTransform()

            # color jit
            color_aug = tf.ColorJitter(
                    (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
            color_aug.get_params((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
            
            for key in list(inputs):
                if "color" in key:
                    raw_img = inputs[key]
                    raw_img = self.fix_crop(raw_img, *self.crop_coords)
                    if is_flip:
                        raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
                    img = self.color_resize(raw_img)
                    img = self.to_tensor(img)
                    aug_img = color_aug(img)
                    inputs[key.replace("_raw", "")] =\
                        self.normalize(img)
                    inputs[key.replace("_raw", "_aug")] =\
                        self.normalize(aug_img)

                elif 'disparity' in key:
                    raw_disp = inputs[key]
                    raw_disp = self.fix_crop(raw_disp, *self.crop_coords)
                    if is_flip:
                        raw_disp = raw_disp.transpose(Image.FLIP_LEFT_RIGHT)
                    disp = self.map_resize(raw_disp)
                    disp = (self.to_tensor(disp) - 1.) / self.depth_scale
                    NaN = disp <= 0
                    disp[NaN] = 1
                    depth_map_gt = baseline * focal_length / disp
                    depth_map_gt[NaN] = 0
                    inputs['depth'] = depth_map_gt
                    if self.mask:
                        mask_depth = MaskGenerator(depth_map_gt, self.mask_ratio, self.mask_patch_size)
                        inputs['mask_depth'] = mask_depth

        else:
            # crop for image
            if self.crop_coords is not None:
                self.fix_crop = tf.functional.crop
            else:
                self.fix_crop = NoneTransform()
            
            # resize
            if self.full_size is not None:
                img_size = self.full_size

                self.color_resize = tf.Resize(img_size, interpolation=Image.ANTIALIAS)
                self.map_resize = tf.Resize(img_size, interpolation=Image.NEAREST)
            else:
                self.color_resize = NoneTransform()
                self.map_resize = NoneTransform()
            
            for key in list(inputs):
                if "color" in key:
                    raw_img = inputs[key]
                    raw_img = self.fix_crop(raw_img, *self.crop_coords)
                    img = self.color_resize(raw_img)
                    img = self.to_tensor(img)
                    inputs[key.replace("_raw", "")] =\
                        self.normalize(img)

                elif 'disparity' in key:
                    raw_disp = inputs[key]
                    raw_disp = self.fix_crop(raw_disp, *self.crop_coords)
                    disp = self.map_resize(raw_disp)
                    disp = (self.to_tensor(disp) - 1.) / self.depth_scale
                    NaN = disp <= 0
                    disp[NaN] = 1
                    depth_map_gt = baseline * focal_length / disp
                    depth_map_gt[NaN] = 0
                    inputs['depth'] = depth_map_gt

        # delete raw data
        inputs.pop("color_s_raw")   
        if not self.load_disp:
            inputs.pop("disparity") 
        inputs["file_info"] = [file_info]

        return inputs

            
    def _get_file_list(self, split_file):
        with open(split_file, 'r') as f:
            files = f.readlines()
            filenames = []
            for f in files:
                file_name = f.replace('\n', '')
                filenames.append(file_name)
        return filenames