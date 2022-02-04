import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt

from transformations import rotate, center_crop, horizontal_flip, vertical_flip, adjust_brightness, adjust_contrast, get_boxes, draw_figures


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root=None, slide_list_filename=None, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self._num_slides, self.img_paths, self.binary_masks_paths, self.label_paths = self.read_slide_list(slide_list_filename)

        self._num_imgs = len(self.img_paths)

        # 'pattern3':1, 'pattern4':2, 'benign':3, 'tangential_benign':4, 
        # 'tangential_malignant':5, 'unknown':6, 'PIN':7, 'artefact':8, 
        # 'malignant':9, 'unknown_checked':10, 'unknown_fp': 11, 'stroma': 12, 
        # 'ignored': 13, 'blood_vessel': 14, 'unidentified': 15, 'others': 16, 'gland': 99,
        self.annotation_label_dict = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 99:1} 

        # print('len(self.imgs): {}'.format(len(self.imgs)))
        # print('self.imgs[:5]: {}'.format(self.imgs[:5]))
        # print('len(self.masks): {}'.format(len(self.masks)))
        # print('self.masks[:5]: {}'.format(self.masks[:5]))
        # print('len(self.labels): {}'.format(len(self.labels)))
        # print('self.labels[:5]: {}'.format(self.labels[:5]))

    @property
    def num_slides(self):
        return self._num_slides

    @property
    def num_imgs(self):
        return self._num_imgs

    def read_slide_list(self, slide_list_filename):
        img_path_list = list()
        binary_masks_path_list = list()
        label_path_list = list()

        slide_id_arr = np.loadtxt(slide_list_filename, comments='#', delimiter='\t', dtype=str)
        slide_id_arr = slide_id_arr.reshape((-1,))
        num_slides = slide_id_arr.shape[0]

        for i in range(num_slides):
            slide_id = slide_id_arr[i]
            # print(slide_id)
        
            img_folder_path = os.path.join(self.root, slide_id, 'img/')
            binary_masks_folder_path = os.path.join(self.root, slide_id, 'binary_masks/')
            label_folder_path = os.path.join(self.root, slide_id, 'label/')
        
            num_imgs = len(list(os.listdir(img_folder_path)))
            # print(num_imgs)
        
            img_path_list += [img_folder_path + str(j) + '.png' for j in range(num_imgs)]
            label_path_list += [label_folder_path + str(j) + '.txt' for j in range(num_imgs)]
            binary_masks_list = os.listdir(binary_masks_folder_path)
            # print(binary_masks_list)
            
            for j in range(num_imgs):
                masks_per_img = [file for file in binary_masks_list if str.startswith(file, str(j) + '__')]
                id_list = [int(file.split('_')[2]) for file in masks_per_img]
                sorted_id_list = np.argsort(id_list)
                sorted_masks_per_img = [binary_masks_folder_path + masks_per_img[id] for id in sorted_id_list]
    
                binary_masks_path_list.append(sorted_masks_per_img)

        return num_slides, img_path_list, binary_masks_path_list, label_path_list


    def __getitem__(self, idx):
        # load images and masks
        img_path = self.img_paths[idx]
        binary_masks_path = self.binary_masks_paths[idx]
        label_path = self.label_paths[idx]
        
        # print(img_path)
        img = Image.open(img_path).convert("RGB")
        binary_masks_list = [Image.open(binary_mask_path) for binary_mask_path in binary_masks_path] # convert binary masks to PIL images
    
        labels = np.loadtxt(label_path, dtype=np.int64, comments='#', delimiter='\t')
        labels = labels.reshape((-1,))
        labels = [self.annotation_label_dict[label] for label in labels]
        labels = np.array(labels)
        labels = labels.reshape((-1,))

        assert labels.shape[0] == len(binary_masks_list), 'length of label list is different from length of binary mask list'
        
                          
        if self.transforms:
            if random.random() > 0.5:
                img, binary_masks_list, labels = horizontal_flip(img, binary_masks_list, labels)
            if random.random() > 0.5:
                img, binary_masks_list, labels = vertical_flip(img, binary_masks_list, labels)   
            img, binary_masks_list, labels = rotate(img, binary_masks_list, labels) 
            img, binary_masks_list, labels = adjust_brightness(img, binary_masks_list, labels) 
            img, binary_masks_list, labels = adjust_contrast(img, binary_masks_list, labels) 
        

        img, binary_masks_list, labels = center_crop(img, binary_masks_list, labels)
        boxes = get_boxes(binary_masks_list)

        assert labels.shape[0] == len(binary_masks_list)

        img = TF.to_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = []
        for mask in binary_masks_list:
            mask = np.array(mask)
            masks.append(mask)
        if len(masks) > 1:    
            masks = np.stack(masks, axis=0)
        elif len(masks) == 1:
            masks = masks[0][np.newaxis, :]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1] + 1) * (boxes[:, 2] - boxes[:, 0] + 1)
        # suppose all instances are not crowd
        num_objs = masks.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)   
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # print(boxes, labels, area)
        # print(masks.shape)

        return img, target

    def __len__(self):
        return len(self.img_paths)
