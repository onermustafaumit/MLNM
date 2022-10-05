import argparse
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from model import Model
from dataset_test_slide_panda import Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


parser = argparse.ArgumentParser(description='Test a trained multi-resolution gland classification model on PANDA dataset.')

parser.add_argument('--init_model_file', default='',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--slide_dir', default='/mnt/Data/PANDA/prostate-cancer-grade-assessment/train_images', help='Image directory', dest='slide_dir')
parser.add_argument('--mask_dir', default='/mnt/Data/PANDA/prostate-cancer-grade-assessment/train_label_masks', help='Image directory', dest='mask_dir')
parser.add_argument('--slide_list_filename', default='', help='slide list test', dest='slide_list_filename')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='2', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--metrics_file', default='test_metrics__panda', help='Text file to write test metrics', dest='metrics_file')

FLAGS = parser.parse_args()
        
# "./saved_models/model_weights__YYYY_MM_DD__HH_MM_SS__XXX.pth"
model_name = FLAGS.init_model_file.split('/')[-1][15:-4]

out_dir = '{}/{}/{}'.format(FLAGS.metrics_file,model_name,FLAGS.slide_list_filename.split('/')[-1][:-4])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

test_metrics_file = '{}/test_scores__{}.txt'.format(out_dir,model_name)
with open(test_metrics_file, 'w') as f:
    f.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
    f.write('# model_name: {}\n'.format(model_name))
    f.write('# slide_list_filename: {}\n'.format(FLAGS.slide_list_filename))
    f.write('# num_classes: {}\n'.format(FLAGS.num_classes))
    f.write('# batch_size: {}\n'.format(FLAGS.batch_size))
    f.write('# metrics_file: {}\n'.format(test_metrics_file))
    f.write('# slide_id\tX_cm\tY_cm\tprediction\tscore_benign\tscore_malignant\n') 


print('model_name: {}'.format(model_name))
print('init_model_file: {}'.format(FLAGS.init_model_file))
print('slide_list_filename: {}'.format(FLAGS.slide_list_filename))
print('num_classes: {}'.format(FLAGS.num_classes))
print('batch_size: {}'.format(FLAGS.batch_size))
print('metrics_file: {}'.format(test_metrics_file))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get the model using our helper function
model = Model(pretrained=False, num_classes=FLAGS.num_classes, num_intermediate_features=64)
# move model to the right device
model.to(device)

if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        state_dict = torch.load(FLAGS.init_model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model_state_dict'])
        print("Model weights loaded successfully from file: ", FLAGS.init_model_file)


print('******************** testing ********************')

# get the slide ids
slide_id_arr = np.loadtxt(FLAGS.slide_list_filename, comments='#', delimiter='\t', dtype=str)
# slide_id_arr = slide_id_arr.reshape((-1,))
slide_id_arr = slide_id_arr[:5,0]
num_slides = slide_id_arr.shape[0]
print('num_slides:{}'.format(num_slides))


model.eval()
with torch.no_grad():

    for s, slide_id in enumerate(slide_id_arr):
        print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

        # dataset for the current patient
        dataset = Dataset(slide_dir=FLAGS.slide_dir, mask_dir=FLAGS.mask_dir, slide_id=slide_id, transforms=False)
        if dataset.num_imgs == 0:
            print('there is no detected glands!!!')

            with open(test_metrics_file, 'a') as f:
                f.write('{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\n'.format(slide_id, 0, 0, 0, 1, 0))

            continue

        num_imgs = dataset.num_imgs
        print("Data - num_imgs: {}".format(num_imgs))

        # define data loaders
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=10)

        
        num_predictions = 0
        running_correct_result = 0

        label_list = []
        predicted_result_list = []
        probs_result_list = []

        pbar = tqdm(total=len(data_loader))
        for i, (img_high, img_medium, img_low, X_cm, Y_cm) in enumerate(data_loader):

            # fig,ax = plt.subplots(1,3)
            # ax[0].imshow(np.transpose(img_high.squeeze().cpu().detach().numpy(), (1,2,0)))
            # ax[1].imshow(np.transpose(img_medium.squeeze().cpu().detach().numpy(), (1,2,0)))
            # ax[2].imshow(np.transpose(img_low.squeeze().cpu().detach().numpy(), (1,2,0)))

            # plt.show()

            img_high, img_medium, img_low = img_high.to(device), img_medium.to(device), img_low.to(device)

            # get logits from the model
            output_high, output_medium, output_low, output_result = model(img_high, img_medium, img_low)

            # obtain probs
            probs_result = F.softmax(output_result, dim=1)

            # obtain predictions
            _, predicted_result = torch.max(output_result, 1)

            predicted_result_arr = predicted_result.cpu().numpy()
            probs_result_arr = probs_result.cpu().numpy()

            temp_num_predictions = predicted_result_arr.shape[0]
            for idx in range(temp_num_predictions):
                with open(test_metrics_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\n'.format(slide_id, X_cm[idx], Y_cm[idx], predicted_result_arr[idx], probs_result_arr[idx, 0], probs_result_arr[idx, 1]))

            pbar.update(1)

        pbar.close()




