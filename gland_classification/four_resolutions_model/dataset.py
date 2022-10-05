import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import random
import torchvision.transforms.functional as TF


class Dataset(torch.utils.data.Dataset):
    """
    Creates a multi-resolution gland classification dataset using cropped patches. 

    Args:
        img_dir_high (str): path to directory storing cropped 40x patches
        img_dir_medium (str): path to directory storing cropped 20x patches
        img_dir_low (str): path to directory storing cropped 10x patches
        img_dir_low2 (str): path to directory storing cropped 5x patches
        slide_list_filename (str): path to text file containing slide ids that 
            will be used to create the dataset
        transforms (bool): flag controlling if some transformations will be applied 
            on the patches for data augmentation. 
            True: apply transformations, False: no transformations
    """
    
    def __init__(self, img_dir_high=None, img_dir_medium=None, img_dir_low=None, img_dir_low2=None, slide_list_filename=None, transforms=False):
        self.img_dir_high = img_dir_high
        self.img_dir_medium = img_dir_medium
        self.img_dir_low = img_dir_low
        self.img_dir_low2 = img_dir_low2
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self._num_slides, self._img_paths_high, self._img_paths_medium, self._img_paths_low, self._img_paths_low2, self._labels = self.read_slide_list(slide_list_filename)

        self._num_imgs = len(self._img_paths_high)

    @property
    def num_slides(self):
        return self._num_slides

    @property
    def num_imgs(self):
        return self._num_imgs

    def __len__(self):
        return len(self._img_paths_high)

    def read_slide_list(self, slide_list_filename):
        img_path_list_high = list()
        img_path_list_medium = list()
        img_path_list_low = list()
        img_path_list_low2 = list()
        label_list = list()

        slide_id_arr = np.loadtxt(slide_list_filename, comments='#', delimiter='\t', dtype=str) #[:2]
        slide_id_arr = slide_id_arr.reshape((-1,))
        num_slides = slide_id_arr.shape[0]
        
        # select annotations that are benign or malignant
        # 'pattern3':1, 'pattern4':2, 'benign':3, 'tangential_benign':4, 
        # 'tangential_malignant':5, 'unknown':6, 'PIN':7, 'artefact':8, 
        # 'malignant':9, 'unknown_checked':10, 'unknown_fp': 11, 'stroma': 12, 
        # 'ignored': 13, 'blood_vessel': 14, 'unidentified': 15, 'others': 16, 'gland': 99,
        benign_malignant_labels = [3, 4, 7, 1, 2, 5, 9]
        labels_dict = {3:0, 4:0, 7:0, 1:1, 2:1, 5:1, 9:1}

        for i in range(num_slides):
            slide_id = slide_id_arr[i]
            # print(slide_id)

            img_folder_path_high = os.path.join(self.img_dir_high, slide_id, 'img/')
            img_folder_path_medium = os.path.join(self.img_dir_medium, slide_id, 'img/')
            img_folder_path_low = os.path.join(self.img_dir_low, slide_id, 'img/')
            img_folder_path_low2 = os.path.join(self.img_dir_low2, slide_id, 'img/')

            label_filename = os.path.join(self.img_dir_high, slide_id, 'cropped_patches_filelist.txt')
            label_arr = np.loadtxt(label_filename, comments = '#', delimiter='\t', dtype=int)
            labels = list(label_arr[:, 5])

            # the ones checked by pathologist
            center_checked_list = list(label_arr[:, -2])
        
            for i, label in enumerate(labels):
                # # keep only the ones checked by pathologist
                # if center_checked_list[i] == 0:
                #     continue

                # # exclude the ones checked by pathologist
                # if center_checked_list[i] == 1:
                #     continue

                if label in benign_malignant_labels:
                    img_path_list_high.append(img_folder_path_high + str(i) + '.png')
                    img_path_list_medium.append(img_folder_path_medium + str(i) + '.png')
                    img_path_list_low.append(img_folder_path_low + str(i) + '.png')
                    img_path_list_low2.append(img_folder_path_low2 + str(i) + '.png')
                    label_list.append(labels_dict[label])

        return num_slides, img_path_list_high, img_path_list_medium, img_path_list_low, img_path_list_low2, label_list
    
    
    def get_transform(self, img_high, img_medium, img_low, img_low2):       

        if random.random() > 0.5:
            img_high = TF.hflip(img_high)
            img_medium = TF.hflip(img_medium)
            img_low = TF.hflip(img_low)
            img_low2 = TF.hflip(img_low2)
        
        if random.random() > 0.5:
            img_high = TF.vflip(img_high)
            img_medium = TF.vflip(img_medium)
            img_low = TF.vflip(img_low)
            img_low2 = TF.vflip(img_low2)


        # random rotation
        angle = random.randint(-180,180)
        img_high = TF.rotate(img_high,angle)
        img_medium = TF.rotate(img_medium,angle)
        img_low = TF.rotate(img_low,angle)
        img_low2 = TF.rotate(img_low2,angle)

        # adjust brightness
        # if random.random() > 0.5:
        brightness_factor = 0.7 + 0.6*random.random()
        img_high = TF.adjust_brightness(img_high,brightness_factor)
        img_medium = TF.adjust_brightness(img_medium,brightness_factor)
        img_low = TF.adjust_brightness(img_low,brightness_factor)
        img_low2 = TF.adjust_brightness(img_low2,brightness_factor)


        # adjust contrast
        # if random.random() > 0.5:
        contrast_factor = 0.7 + 0.6*random.random()
        img_high = TF.adjust_contrast(img_high,contrast_factor)
        img_medium = TF.adjust_contrast(img_medium,contrast_factor)
        img_low = TF.adjust_contrast(img_low,contrast_factor)
        img_low2 = TF.adjust_contrast(img_low2,contrast_factor)

      

            
        return img_high, img_medium, img_low, img_low2

    def __getitem__(self, idx):
        # load images
        img_path_high = self._img_paths_high[idx]
        img_path_medium = self._img_paths_medium[idx]
        img_path_low = self._img_paths_low[idx]
        img_path_low2 = self._img_paths_low2[idx]
        label = self._labels[idx]
        

        # print(img_path)
        img_high = Image.open(img_path_high).convert('RGB')
        img_medium = Image.open(img_path_medium).convert('RGB')
        img_low = Image.open(img_path_low).convert('RGB')
        img_low2 = Image.open(img_path_low2).convert('RGB')
        
        
        if self.transforms:
            img_high, img_medium, img_low, img_low2 = self.get_transform(img_high, img_medium, img_low,img_low2)
        

        img_high = TF.center_crop(img_high,362)
        img_medium = TF.center_crop(img_medium,362)
        img_low = TF.center_crop(img_low,362)
        img_low2 = TF.center_crop(img_low2,362)

        img_high = TF.to_tensor(img_high)
        img_medium = TF.to_tensor(img_medium)
        img_low = TF.to_tensor(img_low)
        img_low2 = TF.to_tensor(img_low2)



        label = torch.as_tensor(label, dtype=torch.int64)
        

        return [img_path_high, img_path_medium, img_path_low, img_path_low2], img_high, img_medium, img_low, img_low2, label



