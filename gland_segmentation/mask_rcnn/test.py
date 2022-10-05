import numpy as np
import argparse
from datetime import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate, evaluate_loss
import utils

from dataset import Dataset


def get_instance_segmentation_model(num_classes, pretrained, pretrained_backbone, trainable_backbone_layers):
    """
    Returns a Mask R-CNN model with specified properties. 

    Args:
        num_classes (int): number of output classes of the model (including the background)
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
  
    Returns:
        Mask R-CNN model

    """
    
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
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

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
        

parser = argparse.ArgumentParser(description='Test a trained Mask R-CNN model segmenting glands in cropped patches')

parser.add_argument('--init_model_file', default='', help='File path of trained Mask R-CNN model', dest='init_model_file')
parser.add_argument('--image_dir', default='../../Images/gland_segmentation/cropped_patches__complete_and_partial_glands_50_50_512/', help='Directory consisting of cropped patches with centred glands', dest='image_dir')
parser.add_argument('--slide_list_filename', default='../dataset/slide_ids_list_gland_segmentation_99_slides_test_saved.txt', help='List of slide ids in training, validation or test set', dest='slide_list_filename')
parser.add_argument('--patch_size', type=int, default=512, help='Patch size', dest='patch_size')
parser.add_argument('--num_classes', type=int, default=1, help='Number of classes', dest='num_classes')
parser.add_argument('--pretrained', type=str2bool, default=False, help=' Use pretrained model on COCO dataset', dest='pretrained')
parser.add_argument('--pretrained_backbone', type=str2bool, default=False, help='Use pretrained ResNet backbone on ImageNet?', dest='pretrained_backbone')
parser.add_argument('--trainable_backbone_layers', type=int, default=5, help='Number of trainable layers in ResNet backbone', dest='trainable_backbone_layers')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size', dest='batch_size')
parser.add_argument('--metrics_dir', type=str, default='saved_metrics', help='Directory where metrics are saved to', dest='metrics_dir')

FLAGS = parser.parse_args()

model_name = ('__').join(FLAGS.init_model_file.split('/')[-1].split('__')[1:])[:-4]

dataset_type = FLAGS.slide_list_filename.split('/')[-1].split('_')[-1].split('.')[0]

FLAGS.metrics_file = os.path.join(FLAGS.metrics_dir, dataset_type + '_metrics__' + model_name + '.txt')

print('init_model_file: {}'.format(FLAGS.init_model_file))
print('image_dir: {}'.format(FLAGS.image_dir))
print('slide_list_filename: {}'.format(FLAGS.slide_list_filename))
print('patch_size: {}'.format(FLAGS.patch_size))
print('num_classes: {}'.format(FLAGS.num_classes))
print('pretrained: {}'.format(FLAGS.pretrained))
print('pretrained_backbone: {}'.format(FLAGS.pretrained_backbone))
print('trainable_backbone_layers: {}'.format(FLAGS.trainable_backbone_layers))
print('batch_size: {}'.format(FLAGS.batch_size))
print('metrics_dir: {}'.format(FLAGS.metrics_dir))
print('metrics_file: {}'.format(FLAGS.metrics_file))

    
if not os.path.exists(FLAGS.metrics_dir):
    os.mkdir(FLAGS.metrics_dir)

dataset = Dataset(root=FLAGS.image_dir, slide_list_filename=FLAGS.slide_list_filename, transforms=False)
num_slides = dataset.num_slides
num_imgs = dataset.num_imgs
print('Data - num_slides: {}'.format(dataset.num_slides))
print('Data - num_imgs: {}'.format(dataset.num_imgs))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = FLAGS.num_classes+1

# get the model using our helper function
model = get_instance_segmentation_model(num_classes, FLAGS.pretrained, FLAGS.pretrained_backbone, FLAGS.trainable_backbone_layers)
# move model to the right device
model.to(device)


if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        model.load_state_dict(torch.load(FLAGS.init_model_file, map_location=lambda storage, loc: storage))
        print('Model weights loaded successfully from file: ', FLAGS.init_model_file)


# print('#################### Testing ####################')

coco_evaluator = evaluate(model, data_loader, device)

bbox_eval = coco_evaluator.coco_eval['bbox'].eval
segm_eval = coco_evaluator.coco_eval['segm'].eval
bbox_str = '\t'.join(['{val:.4f}'.format(val=test) for test in coco_evaluator.coco_eval['bbox'].stats])
segm_str = '\t'.join(['{val:.4f}'.format(val=test) for test in coco_evaluator.coco_eval['segm'].stats])

with open(FLAGS.metrics_file, 'w') as f:
    f.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
    f.write('# image_dir: {}\n'.format(FLAGS.image_dir))
    f.write('# slide_list_filename: {}\n'.format(FLAGS.slide_list_filename))
    f.write('# patch_size: {}\n'.format(FLAGS.patch_size))
    f.write('# num_classes: {}\n'.format(FLAGS.num_classes))
    f.write('# pretrained: {}\n'.format(FLAGS.pretrained))
    f.write('# pretrained_backbone: {}\n'.format(FLAGS.pretrained_backbone))
    f.write('# trainable_backbone_layers: {}\n'.format(FLAGS.trainable_backbone_layers))
    f.write('# batch_size: {}\n'.format(FLAGS.batch_size))
    f.write('# metrics_dir: {}\n'.format(FLAGS.metrics_dir))
    f.write('# metrics_file: {}\n'.format(FLAGS.metrics_file))
    f.write('# bbox_ave_precision_0.50_0.95_all_100\tbbox_ave_precision_0.50_all_100\tbbox_ave_precision_0.75_all_100\tbbox_ave_precision_0.50_0.95_small_100\tbbox_ave_precision_0.50_0.95_medium_100\tbbox_ave_precision_0.50_0.95_large_100\t')
    f.write('bbox_ave_recall_0.50_0.95_all_1\tbbox_ave_recall_0.50_0.95_all_10\tbbox_ave_recall_0.50_0.95_all_100\tbbox_ave_recall_0.50_0.95_small_100\tbbox_ave_recall_0.50_0.95_medium_100\tbbox_ave_recall_0.50_0.95_large_100\tbbox_ave_recall_0.5_all_100\t')
    f.write('segm_ave_precision_0.50_0.95_all_100\tsegm_ave_precision_0.50_all_100\tsegm_ave_precision_0.75_all_100\tsegm_ave_precision_0.50_0.95_small_100\tsegm_ave_precision_0.50_0.95_medium_100\tsegm_ave_precision_0.50_0.95_large_100\t')
    f.write('segm_ave_recall_0.50_0.95_all_1\tsegm_ave_recall_0.50_0.95_all_10\tsegm_ave_recall_0.50_0.95_all_100\tsegm_ave_recall_0.50_0.95_small_100\tsegm_ave_recall_0.50_0.95_medium_100\tsegm_ave_recall_0.50_0.95_large_100\tsegm_ave_recall_0.5_all_100\n')
    f.write('{}\t{:.4f}\t{}\t{:.4f}\n'.format(bbox_str, np.mean(bbox_eval['recall'][0,:,0,2]), segm_str, np.mean(segm_eval['recall'][0,:,0,2])))

