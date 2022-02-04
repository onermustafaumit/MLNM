import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time
from PIL import Image
import imageio

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset_predict_masks import Dataset, collate_fn

import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm


def get_instance_segmentation_model(num_classes, pretrained, pretrained_backbone, trainable_backbone_layers):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained_backbone, trainable_backbone_layers=trainable_backbone_layers)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


annotation_group_color_dict = {'1': [255, 0, 0],
                               '2': [0, 255, 0],
                               '3': [0, 0, 255],
                               '4': [255, 255, 0]}

parser = argparse.ArgumentParser(description='Gland detection at the slide level')

parser.add_argument('--init_model_file', default='', help='File path of trained Mask R-CNN model', dest='init_model_file')
parser.add_argument('--image_dir', default='../../Images/gland_segmentation/cropped_patches__complete_and_partial_glands_50_50_512/', help='Directory consisting of cropped patches with centred glands', dest='image_dir')
parser.add_argument('--slide_list_filename', default='../dataset/slide_ids_list_gland_segmentation_99_slides_test_saved.txt', help='List of slide ids in training, validation or test set', dest='slide_list_filename') #only one patient is being tested
parser.add_argument('--batch_size', type=int, default=8, help='Batch size', dest='batch_size')
parser.add_argument('--num_classes', type=int, default=1, help='Number of classes', dest='num_classes')
parser.add_argument('--pretrained', type=str2bool, default=False, help=' Use pretrained model on COCO dataset', dest='pretrained')
parser.add_argument('--pretrained_backbone', type=str2bool, default=False, help='Use pretrained ResNet backbone on ImageNet?', dest='pretrained_backbone')
parser.add_argument('--trainable_backbone_layers', type=int, default=5, help='Number of trainable layers in ResNet backbone', dest='trainable_backbone_layers')
parser.add_argument('--out_dir', type=str, default='mask_predictions', help='Directory where Mask R-CNN predictions are saved to', dest='out_dir')

FLAGS = parser.parse_args()

model_name = FLAGS.init_model_file.split('/')[-1][15:-4]

dataset_type = FLAGS.slide_list_filename.split('/')[-1].split('_')[-1].split('.')[0]

out_dir = os.path.join(FLAGS.out_dir, model_name, dataset_type)

print('init_model_file: {}'.format(FLAGS.init_model_file))
print('image_dir: {}'.format(FLAGS.image_dir))
print('slide_list_filename: {}'.format(FLAGS.slide_list_filename))
print('out_dir: {}'.format(out_dir))
print('batch_size: {}'.format(FLAGS.batch_size))
print('num_classes: {}'.format(FLAGS.num_classes))

slide_list_filename = FLAGS.slide_list_filename
dataset = Dataset(root=FLAGS.image_dir, slide_list_filename=FLAGS.slide_list_filename)
img_paths = dataset.img_paths
num_slides = dataset.num_slides
num_imgs = dataset.num_imgs
# print("img_paths[:5]: {}".format(img_paths[:5]))
print("num_slides: {}".format(num_slides))
print("num_imgs: {}".format(num_imgs))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

num_classes = FLAGS.num_classes + 1
model = get_instance_segmentation_model(num_classes, FLAGS.pretrained, FLAGS.pretrained_backbone, FLAGS.trainable_backbone_layers)

model.to(device)
model.load_state_dict(torch.load(FLAGS.init_model_file, map_location=lambda storage, loc: storage))
print("Model weights loaded successfully from file: ", FLAGS.init_model_file)

pbar = tqdm(total=len(data_loader))

model.eval()
with torch.no_grad():
    for i, (images, img_path_list) in enumerate(data_loader):

        images = images.to(device)
        prediction = model(images)

        for p,img_path in enumerate(img_path_list):
            splitted_img_path = img_path.split('/')
            img_id = splitted_img_path[-1][:-4]
            slide_id = splitted_img_path[-3]

            outdir = out_dir + '/' + slide_id
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            temp_dict = prediction[p]
            temp_labels = np.array(temp_dict['labels'].cpu())
            temp_scores = np.array(temp_dict['scores'].cpu())
            temp_box_arr = np.array(temp_dict['boxes'].cpu())
            temp_mask_arr = np.array(temp_dict['masks'].cpu())

            np.savez_compressed(outdir + '/{}_predictions.npz'.format(img_id), predicted_labels=temp_labels, predicted_scores=temp_scores, predicted_boxes=temp_box_arr,
                                predicted_masks=temp_mask_arr)



        pbar.update(1)

pbar.close()


