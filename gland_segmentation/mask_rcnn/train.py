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
        

parser = argparse.ArgumentParser(description='Train a Mask R-CNN model to segment glands in cropped patches')

parser.add_argument('--init_model_file', default='',help='File path of Mask R-CNN model to be loaded at the start of training (optional)', dest='init_model_file')
parser.add_argument('--image_dir', default='../../Images/gland_segmentation/cropped_patches__complete_and_partial_glands_50_50_512/', help='Directory consisting of cropped patches with centred glands', dest='image_dir')
parser.add_argument('--slide_list_filename_train', default='../dataset/slide_ids_list_gland_segmentation_99_slides_train_saved.txt', help='List of slide ids in training set', dest='slide_list_filename_train')
parser.add_argument('--slide_list_filename_valid', default='../dataset/slide_ids_list_gland_segmentation_99_slides_valid_saved.txt', help='List of slide ids in validation set', dest='slide_list_filename_valid')
parser.add_argument('--patch_size', type=int, default=512, help='Patch size', dest='patch_size')
parser.add_argument('--num_classes', type=int, default=1, help='Number of classes', dest='num_classes')
parser.add_argument('--pretrained', type=str2bool, default=False, help=' Use pretrained model on COCO dataset?', dest='pretrained')
parser.add_argument('--pretrained_backbone', type=str2bool, default=False, help='Use pretrained ResNet backbone on ImageNet?', dest='pretrained_backbone')
parser.add_argument('--trainable_backbone_layers', type=int, default=5, help='Number of trainable layers in ResNet backbone', dest='trainable_backbone_layers')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size', dest='batch_size')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate', dest='learning_rate')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay', dest='weight_decay')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs', dest='num_epochs')
parser.add_argument('--save_interval', type=int, default=10, help='Model save interval', dest='save_interval')
parser.add_argument('--metrics_dir', type=str, default='saved_metrics', help='Directory where metrics are saved to', dest='metrics_dir')
parser.add_argument('--model_dir', default='saved_models', help='Directory where models are saved to', dest='model_dir')

FLAGS = parser.parse_args()
current_time = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
FLAGS.metrics_train_file = FLAGS.metrics_dir + '/step_train_metrics__' + current_time + '.txt'
FLAGS.metrics_valid_file = FLAGS.metrics_dir + '/step_valid_metrics__' + current_time + '.txt'

print('model_dir: {}'.format(FLAGS.model_dir))
print('init_model_file: {}'.format(FLAGS.init_model_file))
print('image_dir: {}'.format(FLAGS.image_dir))
print('slide_list_filename_train: {}'.format(FLAGS.slide_list_filename_train))
print('slide_list_filename_valid: {}'.format(FLAGS.slide_list_filename_valid))
print('patch_size: {}'.format(FLAGS.patch_size))
print('num_classes: {}'.format(FLAGS.num_classes))
print('pretrained: {}'.format(FLAGS.pretrained))
print('pretrained_backbone: {}'.format(FLAGS.pretrained_backbone))
print('trainable_backbone_layers: {}'.format(FLAGS.trainable_backbone_layers))
print('batch_size: {}'.format(FLAGS.batch_size))
print('learning_rate: {}'.format(FLAGS.learning_rate))
print('weight_decay: {}'.format(FLAGS.weight_decay))
print('num_epochs: {}'.format(FLAGS.num_epochs))
print('save_interval: {}'.format(FLAGS.save_interval))
print('metrics_dir: {}'.format(FLAGS.metrics_dir))
print('metrics_train_file: {}'.format(FLAGS.metrics_train_file))
print('metrics_valid_file: {}'.format(FLAGS.metrics_valid_file))

if not os.path.exists(FLAGS.model_dir):
    os.mkdir(FLAGS.model_dir)
    
if not os.path.exists(FLAGS.metrics_dir):
    os.mkdir(FLAGS.metrics_dir)

train_dataset = Dataset(root=FLAGS.image_dir, slide_list_filename=FLAGS.slide_list_filename_train, transforms=True)
num_slides_train = train_dataset.num_slides
num_imgs_train = train_dataset.num_imgs
print('Training Data - num_slides: {}'.format(train_dataset.num_slides))
print('Training Data - num_imgs: {}'.format(train_dataset.num_imgs))

valid_dataset = Dataset(root=FLAGS.image_dir, slide_list_filename=FLAGS.slide_list_filename_valid, transforms=False)
num_slides_valid = valid_dataset.num_slides
num_imgs_valid = valid_dataset.num_imgs
print('Validation Data - num_slides: {}'.format(valid_dataset.num_slides))
print('Validation Data - num_imgs: {}'.format(valid_dataset.num_imgs))

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1, collate_fn=utils.collate_fn)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and foreground
num_classes = FLAGS.num_classes+1

# get the model using our helper function
model = get_instance_segmentation_model(num_classes, FLAGS.pretrained, FLAGS.pretrained_backbone, FLAGS.trainable_backbone_layers)
# move model to the right device
model.to(device)

if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        model.load_state_dict(torch.load(FLAGS.init_model_file, map_location=lambda storage, loc: storage))
        print('Model weights loaded successfully from file: ', FLAGS.init_model_file)


# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)


with open(FLAGS.metrics_train_file, 'w') as f_train, open(FLAGS.metrics_valid_file, 'w') as f_valid:

    f_train.write('# model_dir: {}\n'.format(FLAGS.model_dir))
    f_train.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
    f_train.write('# image_dir: {}\n'.format(FLAGS.image_dir))
    f_train.write('# slide_list_filename_train: {}\n'.format(FLAGS.slide_list_filename_train))
    f_train.write('# slide_list_filename_valid: {}\n'.format(FLAGS.slide_list_filename_valid))
    f_train.write('# patch_size: {}\n'.format(FLAGS.patch_size))
    f_train.write('# num_classes: {}\n'.format(FLAGS.num_classes))
    f_train.write('# pretrained: {}\n'.format(FLAGS.pretrained))
    f_train.write('# pretrained_backbone: {}\n'.format(FLAGS.pretrained_backbone))
    f_train.write('# trainable_backbone_layers: {}\n'.format(FLAGS.trainable_backbone_layers))
    f_train.write('# batch_size: {}\n'.format(FLAGS.batch_size))
    f_train.write('# learning_rate: {}\n'.format(FLAGS.learning_rate))
    f_train.write('# weight_decay: {}\n'.format(FLAGS.weight_decay))
    f_train.write('# num_epochs: {}\n'.format(FLAGS.num_epochs))
    f_train.write('# save_interval: {}\n'.format(FLAGS.save_interval))
    f_train.write('# metrics_dir: {}\n'.format(FLAGS.metrics_dir))
    f_train.write('# metrics_train_file: {}\n'.format(FLAGS.metrics_train_file))
    f_train.write('# metrics_valid_file: {}\n'.format(FLAGS.metrics_valid_file))
    f_train.write('# epoch\tlr\ttrain_loss\ttrain_loss_classifier\ttrain_loss_box_reg\ttrain_loss_mask\ttrain_loss_objectness\ttrain_loss_rpn_box_reg\n')

    f_valid.write('# epoch\tbbox_ave_precision_0.50_0.95_all_100\tbbox_ave_precision_0.50_all_100\tbbox_ave_precision_0.75_all_100\tbbox_ave_precision_0.50_0.95_small_100\tbbox_ave_precision_0.50_0.95_medium_100\tbbox_ave_precision_0.50_0.95_large_100\t')
    f_valid.write('bbox_ave_recall_0.50_0.95_all_1\tbbox_ave_recall_0.50_0.95_all_10\tbbox_ave_recall_0.50_0.95_all_100\tbbox_ave_recall_0.50_0.95_small_100\tbbox_ave_recall_0.50_0.95_medium_100\tbbox_ave_recall_0.50_0.95_large_100\tbbox_ave_recall_0.5_all_100\t')
    f_valid.write('segm_ave_precision_0.50_0.95_all_100\tsegm_ave_precision_0.50_all_100\tsegm_ave_precision_0.75_all_100\tsegm_ave_precision_0.50_0.95_small_100\tsegm_ave_precision_0.50_0.95_medium_100\tsegm_ave_precision_0.50_0.95_large_100\t')
    f_valid.write('segm_ave_recall_0.50_0.95_all_1\tsegm_ave_recall_0.50_0.95_all_10\tsegm_ave_recall_0.50_0.95_all_100\tsegm_ave_recall_0.50_0.95_small_100\tsegm_ave_recall_0.50_0.95_medium_100\tsegm_ave_recall_0.50_0.95_large_100\tsegm_ave_recall_0.5_all_100\n')

for epoch in range(FLAGS.num_epochs):
    # train for one epoch, printing every 10 iterations
    metric_logger_train = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        
    with open(FLAGS.metrics_train_file, 'a') as f_train:
        f_train.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(str(epoch+1),
                                                                str(metric_logger_train.meters['lr']),
                                                                str(metric_logger_train.meters['loss']),
                                                                str(metric_logger_train.meters['loss_classifier']),
                                                                str(metric_logger_train.meters['loss_box_reg']),
                                                                str(metric_logger_train.meters['loss_mask']),
                                                                str(metric_logger_train.meters['loss_objectness']),
                                                                str(metric_logger_train.meters['loss_rpn_box_reg'])))
        
    if (epoch+1) % 10 == 0:   
        coco_evaluator = evaluate(model, data_loader_valid, device)
        bbox_eval = coco_evaluator.coco_eval['bbox'].eval
        segm_eval = coco_evaluator.coco_eval['segm'].eval 
        
        bbox_str = '\t'.join(['{val:.4f}'.format(val=val) for val in coco_evaluator.coco_eval['bbox'].stats])
        segm_str = '\t'.join(['{val:.4f}'.format(val=val) for val in coco_evaluator.coco_eval['segm'].stats])
        
        with open(FLAGS.metrics_valid_file, 'a') as f_valid:
            f_valid.write('{}\t{}\t{:.4f}\t{}\t{:.4f}\n'.format(epoch+1, bbox_str, np.mean(bbox_eval['recall'][0,:,0,2]), segm_str, np.mean(segm_eval['recall'][0,:,0,2])))
           
    if (epoch+1) % FLAGS.save_interval == 0:
        model_weights_filename = os.path.join(FLAGS.model_dir, 'model_weights__' + current_time + '__' + str(epoch+1) + '.pth')
        state_dict = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(model.state_dict(), model_weights_filename)
        print("Model weights saved in file: ", model_weights_filename)
        

