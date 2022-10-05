import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms.functional as TF

import openslide
import xml.etree.ElementTree as ET


class Dataset(torch.utils.data.Dataset):
    """
    Creates a multi-resolution gland classification dataset for inference 
    on a slide in the PANDA dataset. 

    Args:
        slide_dir (str): path to directory storing H&E slides
        mask_dir (str): path to directory storing mask slides
        slide_id (str): slide id to create the dataset
        transforms (bool): flag controlling if some transformations will be applied 
            on the patches for data augmentation during inference. 
            True: apply transformations, False: no transformations
    """
    
    def __init__(self, slide_dir=None, mask_dir=None, slide_id=None, transforms=False):
        self.slide_dir = slide_dir
        self.mask_dir = mask_dir
        self.slide_id = slide_id
        self.transforms = transforms

        self.slide_path = '{}/{}.tiff'.format(slide_dir,slide_id)
        self.mask_path = '{}/{}_mask.tiff'.format(mask_dir,slide_id)

        self.coordinates = self.get_patch_coordinates()

        self._num_imgs = len(self.coordinates)


    @property
    def num_imgs(self):
        return self._num_imgs
    
    def __len__(self):
        return self._num_imgs


    def get_patch_coordinates(self):
        # to store valid patch coordinates
        coordinates_list = []

        # we will read mask at level5
        # then we will return a patch coordinate
        # and a patch label
        slide = openslide.OpenSlide(self.mask_path)

        img_read_level = slide.level_count - 1
        img_size = slide.level_dimensions[img_read_level]
        img = slide.read_region((0, 0), img_read_level, img_size)
        img_arr = np.array(img)[:,:,0]

        slide.close()


        # 0: background (non tissue) or unknown
        # 1: stroma (connective tissue, non-epithelium tissue)
        # 2: healthy (benign) epithelium
        # 3: cancerous epithelium (Gleason 3)
        # 4: cancerous epithelium (Gleason 4)
        # 5: cancerous epithelium (Gleason 5)


        # sliding window over the mask image
        # non-overlapping patches: stride=32
        stride = 32
        mask_height, mask_width = img_arr.shape
        r_steps = mask_height // stride
        c_steps = mask_width // stride
        for r in range(r_steps):
            for c in range(c_steps):
                y = r*stride
                x = c*stride
                
                temp_mask = img_arr[y:y+stride,x:x+stride]

                # if there are foreground pixels (100+ out of 1024)
                # add coordinates to coordinates_list
                if np.sum(temp_mask) > 100:
                    x_cm = (x+16)*16
                    y_cm = (y+16)*16
                    coordinates_list.append((x_cm,y_cm))

        return coordinates_list


    def get_transform(self, img_high, img_medium, img_low):       

        if random.random() > 0.5:
            img_high = TF.hflip(img_high)
            img_medium = TF.hflip(img_medium)
            img_low = TF.hflip(img_low)
        
        if random.random() > 0.5:
            img_high = TF.vflip(img_high)
            img_medium = TF.vflip(img_medium)
            img_low = TF.vflip(img_low)


        # random rotation
        angle = random.randint(-180,180)
        img_high = TF.rotate(img_high,angle)
        img_medium = TF.rotate(img_medium,angle)
        img_low = TF.rotate(img_low,angle)

        # adjust brightness
        # if random.random() > 0.5:
        brightness_factor = 0.7 + 0.6*random.random()
        img_high = TF.adjust_brightness(img_high,brightness_factor)
        img_medium = TF.adjust_brightness(img_medium,brightness_factor)
        img_low = TF.adjust_brightness(img_low,brightness_factor)


        # adjust contrast
        # if random.random() > 0.5:
        contrast_factor = 0.7 + 0.6*random.random()
        img_high = TF.adjust_contrast(img_high,contrast_factor)
        img_medium = TF.adjust_contrast(img_medium,contrast_factor)
        img_low = TF.adjust_contrast(img_low,contrast_factor)


        return img_high, img_medium, img_low


    def get_images(self, X_cm, Y_cm):
    
        # we will read an image with size 2048 at level0
        # then we will downsample and crop the center for three resolutions
        im_read_level = 0
        im_read_size = (2048,2048)
        X_min_frame = X_cm - int(im_read_size[0]/2)
        Y_min_frame = Y_cm - int(im_read_size[1]/2)

        slide = openslide.OpenSlide(self.slide_path)
        img = slide.read_region((X_min_frame, Y_min_frame), im_read_level, im_read_size)
        img = img.convert('RGB')
        img_arr = np.array(img)
        slide.close()

        img_high = img.crop((768,768,1280,1280))

        img_medium = img.crop((512,512,1536,1536))
        img_medium = img_medium.resize((512,512),Image.ANTIALIAS)

        img_low = img.resize((512,512),Image.ANTIALIAS)

        return img_high, img_medium, img_low


    def __getitem__(self, idx):
        X_cm, Y_cm = self.coordinates[idx]

        img_high, img_medium, img_low = self.get_images(X_cm,Y_cm)
        

        if self.transforms:
            img_high, img_medium, img_low = self.get_transform(img_high, img_medium, img_low)
        

        img_high = TF.center_crop(img_high,362)
        img_medium = TF.center_crop(img_medium,362)
        img_low = TF.center_crop(img_low,362)

        img_high = TF.to_tensor(img_high)
        img_medium = TF.to_tensor(img_medium)
        img_low = TF.to_tensor(img_low)
        

        return img_high, img_medium, img_low, X_cm, Y_cm

