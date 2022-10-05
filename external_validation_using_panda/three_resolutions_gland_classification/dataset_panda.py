import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms.functional as TF

import openslide

class Dataset(torch.utils.data.Dataset):
    """
    Creates a multi-resolution gland classification dataset using 
    slides and masks in the PANDA dataset. 

    Args:
        slide_dir (str): path to directory storing H&E slides
        mask_dir (str): path to directory storing mask slides
        slide_list_filename (str): path to text file containing slide ids that 
            will be used to create the dataset
        transforms (bool): flag controlling if some transformations will be applied 
            on the patches for data augmentation. 
            True: apply transformations, False: no transformations
    """

    def __init__(self, slide_dir=None, mask_dir=None, slide_list_filename=None, transforms=False):
        self.slide_dir = slide_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        self._slide_ids, self._slide_labels, self._data_providers = self.read_slide_list(slide_list_filename)

        self._num_slides = len(self._slide_ids)


        ### source - radboud ###
        # 0: background (non tissue) or unknown
        # 1: stroma (connective tissue, non-epithelium tissue)
        # 2: healthy (benign) epithelium
        # 3: cancerous epithelium (Gleason 3)
        # 4: cancerous epithelium (Gleason 4)
        # 5: cancerous epithelium (Gleason 5)
        ### target ###
        # 0 - benign
        # 1 - malignant
        self._label_dict_radboud =  {   
                                        0:0,
                                        1:0,
                                        2:0,
                                        3:1,
                                        4:1,
                                        5:1
                                    }

        ### source - karolinska ###
        # 0: background (non tissue) or unknown
        # 1: benign tissue (stroma and epithelium combined)
        # 2: cancerous tissue (stroma and epithelium combined)
        ### target ###
        # 0 - benign
        # 1 - malignant
        self._label_dict_karolinska =  {   
                                        0:0,
                                        1:0,
                                        2:1
                                    }

    @property
    def num_slides(self):
        return self._num_slides


    def __len__(self):
        return self._num_slides

    def read_slide_list(self, slide_list_filename):
        data_arr = np.loadtxt(slide_list_filename, delimiter='\t', comments='#', dtype=str)
        slide_ids = data_arr[:,0]
        data_providers = data_arr[:,1]
        isup_grades = np.asarray(data_arr[:,2], dtype=int)
        # gleason_scores = data_arr[:,3]

        labels = np.asarray(isup_grades > 0, dtype=int)

        return slide_ids, labels, data_providers
    
    
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


    def get_images(self, slide_path, X_cm, Y_cm):
    
        # we will read an image with size 2048 at level0
        # then we will downsample and crop the center for three resolutions
        im_read_level = 0
        im_read_size = (2048,2048)
        X_min_frame = X_cm - int(1024)
        Y_min_frame = Y_cm - int(1024)

        slide = openslide.OpenSlide(slide_path)
        img = slide.read_region((X_min_frame, Y_min_frame), im_read_level, im_read_size)
        slide.close()

        img = img.convert('RGB')
        img_arr = np.array(img)

        img_high = img.crop((768,768,1280,1280))

        img_medium = img.crop((512,512,1536,1536))
        img_medium = img_medium.resize((512,512),Image.ANTIALIAS)

        img_low = img.resize((512,512),Image.ANTIALIAS)

        return img_high, img_medium, img_low

    def get_patch_center(self, mask_path, data_provider):
    
        # we will read mask at level5
        # then we will return a patch coordinate
        # and a patch label
        slide = openslide.OpenSlide(mask_path)

        img_read_level = slide.level_count - 1
        img_size = slide.level_dimensions[img_read_level]
        img = slide.read_region((0, 0), img_read_level, img_size)
        img_arr = np.array(img)[:,:,0]

        

        # find all foreground pixels - radboud
        # 0: background (non tissue) or unknown
        # 1: stroma (connective tissue, non-epithelium tissue)
        # 2: healthy (benign) epithelium
        # 3: cancerous epithelium (Gleason 3)
        # 4: cancerous epithelium (Gleason 4)
        # 5: cancerous epithelium (Gleason 5)

        ### find all foreground pixels - karolinska
        # 0: background (non tissue) or unknown
        # 1: benign tissue (stroma and epithelium combined)
        # 2: cancerous tissue (stroma and epithelium combined)

        # print(np.unique(img_arr))
        indices = np.where(img_arr>0)
        num_pixels = len(indices[0])
        random_ind = np.random.randint(num_pixels)
        x = indices[1][random_ind]
        y = indices[0][random_ind]

        x_cm = 16*x
        y_cm = 16*y

        x_left = x_cm - 256
        y_top = y_cm - 256
        # print('x_left: {}, y_top: {}'.format(x_left,y_top))

        img_read_level = 0
        img_size = (512,512)
        img = slide.read_region((x_left, y_top), img_read_level, img_size)
        img_arr = np.array(img)[:,:,0]
        # print('img_arr.shape: {}'.format(img_arr.shape))

        labels = np.unique(img_arr)
        # print('labels:{}'.format(labels))
        if data_provider == 'radboud':
            label = self._label_dict_radboud[np.amax(labels)]
        else:
            label = self._label_dict_karolinska[np.amax(labels)]

        # print('label:{}'.format(label))
        
        slide.close()
        
        return x_cm, y_cm, label
        


    def __getitem__(self, idx):
        # load images
        slide_id = self._slide_ids[idx]
        slide_label = self._slide_labels[idx]
        slide_data_provider = self._data_providers[idx]
        
        slide_path = '{}/{}.tiff'.format(self.slide_dir,slide_id)
        mask_path = '{}/{}_mask.tiff'.format(self.mask_dir,slide_id)

        X_cm,Y_cm,patch_label = self.get_patch_center(mask_path,slide_data_provider)

        img_high, img_medium, img_low = self.get_images(slide_path,X_cm,Y_cm)
        

        if self.transforms:
            img_high, img_medium, img_low = self.get_transform(img_high, img_medium, img_low)
        

        img_high = TF.center_crop(img_high,362)
        img_medium = TF.center_crop(img_medium,362)
        img_low = TF.center_crop(img_low,362)

        img_high = TF.to_tensor(img_high)
        img_medium = TF.to_tensor(img_medium)
        img_low = TF.to_tensor(img_low)


        patch_label = torch.as_tensor(patch_label, dtype=torch.int64)
        

        return img_high, img_medium, img_low, patch_label



