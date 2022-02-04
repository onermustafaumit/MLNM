import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

    
def rotate(image, binary_masks, labels):
    angle = random.randint(-180, 180)
    image_new = TF.rotate(image, angle)
    labels_new = labels.copy()
    
    binary_masks_new = [TF.rotate(binary_mask, angle) for binary_mask in binary_masks]
    for i in range(len(binary_masks_new)-1 , -1, -1):
        if np.array(binary_masks_new[i]).sum() == 0: # no gland in mask
            del binary_masks_new[i]
            labels_new = np.delete(labels_new, [i])   
            
    return image_new, binary_masks_new, labels_new
        
def center_crop(image, binary_masks, labels):
    width, height = image.size
    output_size = [int(1/math.sqrt(2)*height), int(1/math.sqrt(2)*width)] # image.shape: (channels, height, width)
    image_new = TF.center_crop(image, output_size) # output_size: (height, width)
    labels_new = labels.copy()
    binary_masks_new = [TF.center_crop(binary_mask, output_size) for binary_mask in binary_masks]   
    for i in range(len(binary_masks_new)-1, -1, -1):
        if np.array(binary_masks_new[i]).sum() == 0: # no gland in mask
            del binary_masks_new[i]
            labels_new = np.delete(labels_new, [i])          
    
    return image_new, binary_masks_new, labels_new

                
def horizontal_flip(image, binary_masks, labels):
    image_new = TF.hflip(image)
    binary_masks_new = [TF.hflip(mask) for mask in binary_masks]

    return image_new, binary_masks_new, labels

def vertical_flip(image, binary_masks, labels):
    image_new = TF.vflip(image)
    binary_masks_new = [TF.vflip(mask) for mask in binary_masks] 
                        
    return image_new, binary_masks_new, labels

def adjust_brightness(image, binary_masks, labels):
    brightness_factor = 0.7+0.6*random.random()
    image_new = TF.adjust_brightness(image, brightness_factor) 
    
    return image_new, binary_masks, labels

def adjust_contrast(image, binary_masks, labels):
    contrast_factor = 0.7+0.6*random.random()
    image_new = TF.adjust_contrast(image, contrast_factor) 
    
    return image_new, binary_masks, labels
    
        
def get_boxes(binary_masks):
    boxes = []
    for mask in binary_masks:
        mask = np.array(mask)
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
            
        if xmin == xmax:
            if xmin > 0:
                xmin = xmin - 1
            else:
                xmax = xmax + 1
        if ymin == ymax:
            if ymin > 0:
                ymin = ymin - 1
            else:
                ymax = ymax + 1
                
        boxes.append(np.array([xmin, ymin, xmax, ymax]))
    if len(boxes) > 1:    
        boxes = np.stack(boxes, axis=0)
    elif len(boxes) == 1:
        boxes = boxes[0][np.newaxis, :]

    return boxes
        
def draw_figures(img, boxes, masks, title):
    if len(masks) > 1:
        masks = np.stack(masks, axis=0)
    elif len(masks) == 1:
        masks = masks[0][np.newaxis, :]
    masks_add = np.sum(masks, axis=0)
    masks_add[masks_add > 1] = 1
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(img)
    for [xmin, ymin, xmax, ymax] in boxes:
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)

    ax[2].imshow(masks_add)
    fig.suptitle(title)
    plt.tight_layout()
