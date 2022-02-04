import argparse
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from model import Model
from dataset import Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches into different genetic ITH groups')

parser.add_argument('--init_model_file', default='',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--image_dir_high', default='../../Images/cropped_patches__complete_and_partial_glands__50__25__512', help='Image directory', dest='image_dir_high')
parser.add_argument('--image_dir_medium', default='../../Images/cropped_patches__complete_and_partial_glands__50__50__512', help='Image directory', dest='image_dir_medium')
parser.add_argument('--image_dir_low', default='../../Images/cropped_patches__complete_and_partial_glands__50__100__512', help='Image directory', dest='image_dir_low')
parser.add_argument('--image_dir_low2', default='../../Images/cropped_patches__complete_and_partial_glands__50__200__512', help='Image directory', dest='image_dir_low2')
parser.add_argument('--slide_list_filename_test', default='../dataset/slide_ids_list_gland_classification_test.txt', help='slide list test', dest='slide_list_filename_test')
parser.add_argument('--dataset_type', default='test', help='', dest='dataset_type')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='32', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--metrics_file', default='test_metrics', help='Text file to write test metrics', dest='metrics_file')

FLAGS = parser.parse_args()


model_name = FLAGS.init_model_file.split('/')[-1][15:-4]

out_dir = '{}/{}/{}'.format(FLAGS.metrics_file,model_name,FLAGS.dataset_type)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

test_metrics_file = '{}/test_scores__{}.txt'.format(out_dir,model_name)
with open(test_metrics_file, 'w') as f:
    f.write('# model_name: {}\n'.format(model_name))
    f.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
    f.write('# image_dir_high: {}\n'.format(FLAGS.image_dir_high))
    f.write('# image_dir_medium: {}\n'.format(FLAGS.image_dir_medium))
    f.write('# image_dir_low: {}\n'.format(FLAGS.image_dir_low))
    f.write('# image_dir_low2: {}\n'.format(FLAGS.image_dir_low2))
    f.write('# slide_list_filename_test: {}\n'.format(FLAGS.slide_list_filename_test))
    f.write('# num_classes: {}\n'.format(FLAGS.num_classes))
    f.write('# batch_size: {}\n'.format(FLAGS.batch_size))
    f.write('# metrics_file: {}\n'.format(test_metrics_file))
    f.write('# patient_id\tslide_id\timage_id\tlabel\tprediction\tscore_benign\tscore_malignant\n') 


print('model_name: {}'.format(model_name))
print('init_model_file: {}'.format(FLAGS.init_model_file))
print('image_dir_high: {}'.format(FLAGS.image_dir_high))
print('image_dir_medium: {}'.format(FLAGS.image_dir_medium))
print('image_dir_low: {}'.format(FLAGS.image_dir_low))
print('image_dir_low2: {}'.format(FLAGS.image_dir_low2))
print('slide_list_filename_test: {}'.format(FLAGS.slide_list_filename_test))
print('num_classes: {}'.format(FLAGS.num_classes))
print('batch_size: {}'.format(FLAGS.batch_size))
print('metrics_file: {}'.format(test_metrics_file))


test_dataset = Dataset(img_dir_high=FLAGS.image_dir_high, img_dir_medium=FLAGS.image_dir_medium, img_dir_low=FLAGS.image_dir_low, img_dir_low2=FLAGS.image_dir_low2, slide_list_filename=FLAGS.slide_list_filename_test, transforms=False)
num_imgs_test = test_dataset.num_imgs
print("Test Data - num_imgs: {}".format(test_dataset.num_imgs))

# define data loaders
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1)

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

pbar = tqdm(total=len(data_loader_test))

num_predictions = 0
running_correct_result = 0

label_list = []
predicted_result_list = []
probs_result_list = []

model.eval()
with torch.no_grad():
    for i, (img_paths, img_high, img_medium, img_low, img_low2, label) in enumerate(data_loader_test):

        img_high, img_medium, img_low, img_low2, label = img_high.to(device), img_medium.to(device), img_low.to(device), img_low2.to(device), label.to(device)

        # get logits from the model
        output_high, output_medium, output_low, output_low2, output_result = model(img_high, img_medium, img_low, img_low2)

        # obtain probs
        probs_result = F.softmax(output_result, dim=1)

        # obtain predictions
        _, predicted_result = torch.max(output_result, 1)

        correct_result = (predicted_result == label).sum().item()
        running_correct_result += correct_result

        label_arr = label.cpu().numpy()
        predicted_result_arr = predicted_result.cpu().numpy()
        probs_result_arr = probs_result.cpu().numpy()

        temp_num_predictions = label_arr.shape[0]
        num_predictions += temp_num_predictions
        label_list += list(label_arr)
        predicted_result_list += list(predicted_result_arr)
        probs_result_list += list(probs_result_arr)

        for idx in range(temp_num_predictions):
            with open(test_metrics_file, 'a') as f:
                temp_img_path = img_paths[0][idx]
                patient_id = temp_img_path.split('/')[-3].split('_')[1]
                slide_id = temp_img_path.split('/')[-3].split('_')[3]
                img_id = temp_img_path.split('/')[-1].split('.')[0]
                f.write('{}\t{}\t{}\t{}\t{}\t{:0.4f}\t{:.4f}\n'.format(patient_id, slide_id, img_id, label_arr[idx], predicted_result_arr[idx], probs_result_arr[idx, 0], probs_result_arr[idx, 1]))


        pbar.update(1)

pbar.close()




test_acc_result = running_correct_result / num_predictions
print('test_acc_result:{:.4f}'.format(test_acc_result))

# confusion matrix
cm_test = confusion_matrix(label_list, predicted_result_list, labels=[0,1])
print('cm_test:{}'.format(cm_test))

# per-class accuracy: TPR and TNR
class_acc_test = cm_test.diagonal() / cm_test.sum(1)
print('TNR:{:.4f}, TPR:{:.4f}'.format(class_acc_test[0],class_acc_test[1]))

# Receiver operating chracteristic curve and area under curve value
label_arr = np.array(label_list)
probs_result_arr = np.vstack(probs_result_list)

fpr, tpr, _ = roc_curve(label_arr, probs_result_arr[:,1])
roc_auc = auc(fpr, tpr)


test_metrics_summary_file = '{}/test_metrics_summary__{}.txt'.format(out_dir,model_name)
with open(test_metrics_summary_file, 'w') as f:
    f.write('# model_name: {}\n'.format(model_name))
    f.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
    f.write('# image_dir_high: {}\n'.format(FLAGS.image_dir_high))
    f.write('# image_dir_medium: {}\n'.format(FLAGS.image_dir_medium))
    f.write('# image_dir_low: {}\n'.format(FLAGS.image_dir_low))
    f.write('# image_dir_low2: {}\n'.format(FLAGS.image_dir_low2))
    f.write('# slide_list_filename_test: {}\n'.format(FLAGS.slide_list_filename_test))
    f.write('# num_classes: {}\n'.format(FLAGS.num_classes))
    f.write('# batch_size: {}\n'.format(FLAGS.batch_size))
    f.write('# test_metrics_summary_file: {}\n'.format(test_metrics_summary_file))
    f.write('# test_acc_result\n')
    f.write('{:.4f}\n'.format(test_acc_result))
    f.write('# cm_test: cm_test[0,0]\tcm_test[0,1]\tcm_test[1,0]\tcm_test[1,1]\n')
    f.write('{:d}\t{:d}\t{:d}\t{:d}\n'.format(cm_test[0,0],cm_test[0,1],cm_test[1,0],cm_test[1,1]))
    f.write('# TNR\tTPR\n')
    f.write('{:.4f}\t{:.4f}\n'.format(class_acc_test[0],class_acc_test[1]))
    f.write('# roc_auc\n')
    f.write('{:.4f}\n'.format(roc_auc))


plt.rcParams.update({'font.size':12,'axes.labelsize':12})

fig,ax = plt.subplots(figsize=(3,3))
lw = 2
ax.plot(fpr, tpr, color='darkorange', lw=lw, label='AUROC = %0.2f' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
ax.set_xlim([-0.05, 1.05])
ax.set_xticks(np.arange(0,1.05,0.2))
ax.set_ylim([-0.05, 1.05])
ax.set_yticks(np.arange(0,1.05,0.2))
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('AUROC = %0.4f' % roc_auc)
# ax.legend(loc='lower right')
ax.grid()

fig.tight_layout()
fig_filename = '{}/ROC__{}.png'.format(out_dir,model_name)
fig.savefig(fig_filename, dpi=200)

fig_filename = '{}/ROC__{}.pdf'.format(out_dir,model_name)
fig.savefig(fig_filename, dpi=200)

# plt.show()
           
plt.close('all')

