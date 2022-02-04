import argparse
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.utils.data

from model import Model
from dataset import Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches into different genetic ITH groups')


parser.add_argument('--model_dir', default='saved_models/', help='Directory to save models', dest='model_dir')
parser.add_argument('--init_model_file', default='',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--image_dir_high', default='../../Images/gland_classification/cropped_patches__complete_and_partial_glands_50_25_512', help='Image directory', dest='image_dir_high')
parser.add_argument('--image_dir_medium', default='../../Images/gland_classification/cropped_patches__complete_and_partial_glands_50_50_512', help='Image directory', dest='image_dir_medium')
parser.add_argument('--image_dir_low', default='../../Images/gland_classification/cropped_patches__complete_and_partial_glands_50_100_512', help='Image directory', dest='image_dir_low')
parser.add_argument('--image_dir_low2', default='../../Images/gland_classification/cropped_patches__complete_and_partial_glands_50_200_512', help='Image directory', dest='image_dir_low2')
parser.add_argument('--slide_list_filename_train', default='../dataset/slide_ids_list_gland_classification_46_slides_train_saved.txt', help='slide list train', dest='slide_list_filename_train')
parser.add_argument('--slide_list_filename_valid', default='../dataset/slide_ids_list_gland_classification_46_slides_valid_saved.txt', help='slide list valid', dest='slide_list_filename_valid')
parser.add_argument('--slide_list_filename_test', default='../dataset/slide_ids_list_gland_classification_46_slides_test_saved.txt', help='slide list test', dest='slide_list_filename_test')
parser.add_argument('--patch_size', default='512', type=int, help='Patch size', dest='patch_size')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--pretrained', default=False, help='Pretrain model on ImageNet', dest='pretrained')
parser.add_argument('--batch_size', default='16', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--learning_rate', default='5e-4', type=float, help='Learning rate', dest='learning_rate')
parser.add_argument('--weight_decay', default='5e-5', type=float, help='Weight decay', dest='weight_decay')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs', dest='num_epochs')
parser.add_argument('--save_interval', default=10, type=int, help='Model save interval (default: 1000)', dest='save_interval')
parser.add_argument('--metrics_file', default='saved_metrics', help='Text file to write step, loss, accuracy metrics', dest='metrics_file')

FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
    
if not os.path.exists(FLAGS.metrics_file):
    os.makedirs(FLAGS.metrics_file)
        
current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")
FLAGS.metrics_loss_file = FLAGS.metrics_file + '/step_loss_metrics' + current_time + '.txt'
FLAGS.metrics_acc_file = FLAGS.metrics_file + '/step_acc_metrics' + current_time + '.txt'
FLAGS.metrics_cm_file = FLAGS.metrics_file + '/step_confusion_matrices' + current_time + '.txt'

FLAGS.test_loss_file = FLAGS.metrics_file + '/test_loss_metrics' + current_time + '.txt'
FLAGS.test_acc_file = FLAGS.metrics_file + '/test_acc_metrics' + current_time + '.txt'
FLAGS.test_cm_file = FLAGS.metrics_file + '/test_confusion_matrices' + current_time + '.txt'

print('current_time: {}'.format(current_time))
print('model_dir: {}'.format(FLAGS.model_dir))
print('init_model_file: {}'.format(FLAGS.init_model_file))
print('image_dir_high: {}'.format(FLAGS.image_dir_high))
print('image_dir_medium: {}'.format(FLAGS.image_dir_medium))
print('image_dir_low: {}'.format(FLAGS.image_dir_low))
print('image_dir_low2: {}'.format(FLAGS.image_dir_low2))
print('slide_list_filename_train: {}'.format(FLAGS.slide_list_filename_train))
print('slide_list_filename_valid: {}'.format(FLAGS.slide_list_filename_valid))
print('slide_list_filename_train: {}'.format(FLAGS.slide_list_filename_train))
print('patch_size: {}'.format(FLAGS.patch_size))
print('num_classes: {}'.format(FLAGS.num_classes))
print('pretrained: {}'.format(FLAGS.pretrained))
print('batch_size: {}'.format(FLAGS.batch_size))
print('learning_rate: {}'.format(FLAGS.learning_rate))
print('weight_decay: {}'.format(FLAGS.weight_decay))
print('num_epochs: {}'.format(FLAGS.num_epochs))
print('save_interval: {}'.format(FLAGS.save_interval))
print('metrics_file: {}'.format(FLAGS.metrics_file))
print('# metrics_loss_file: {}'.format(FLAGS.metrics_loss_file))
print('# metrics_acc_file: {}'.format(FLAGS.metrics_acc_file))
print('# metrics_cm_file: {}'.format(FLAGS.metrics_cm_file))
print('# test_loss_file: {}'.format(FLAGS.test_loss_file))
print('# test_acc_file: {}'.format(FLAGS.test_acc_file))
print('# test_cm_file: {}'.format(FLAGS.test_cm_file))   

train_dataset = Dataset(img_dir_high=FLAGS.image_dir_high, img_dir_medium=FLAGS.image_dir_medium, img_dir_low=FLAGS.image_dir_low, img_dir_low2=FLAGS.image_dir_low2, slide_list_filename=FLAGS.slide_list_filename_train, transforms=True)
num_imgs_train = train_dataset.num_imgs
print("Training Data - num_imgs: {}".format(train_dataset.num_imgs))

valid_dataset = Dataset(img_dir_high=FLAGS.image_dir_high, img_dir_medium=FLAGS.image_dir_medium, img_dir_low=FLAGS.image_dir_low, img_dir_low2=FLAGS.image_dir_low2, slide_list_filename=FLAGS.slide_list_filename_valid, transforms=False)
num_imgs_valid = valid_dataset.num_imgs
print("Validation Data - num_imgs: {}".format(valid_dataset.num_imgs))

test_dataset = Dataset(img_dir_high=FLAGS.image_dir_high, img_dir_medium=FLAGS.image_dir_medium, img_dir_low=FLAGS.image_dir_low, img_dir_low2=FLAGS.image_dir_low2, slide_list_filename=FLAGS.slide_list_filename_test, transforms=False)
num_imgs_test = test_dataset.num_imgs
print("Test Data - num_imgs: {}".format(test_dataset.num_imgs))

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get the model using our helper function
model = Model(FLAGS.pretrained, FLAGS.num_classes, num_intermediate_features=64)
# move model to the right device
model.to(device)

# define criterion
criterion = nn.CrossEntropyLoss()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)

if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        state_dict = torch.load(FLAGS.init_model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        print("Model weights loaded successfully from file: ", FLAGS.init_model_file)


with open(FLAGS.metrics_loss_file, 'w') as f:
    f.write('# current_time: {}\n'.format(current_time))
    f.write('# model_dir: {}\n'.format(FLAGS.model_dir))
    f.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
    f.write('# image_dir_high: {}\n'.format(FLAGS.image_dir_high))
    f.write('# image_dir_medium: {}\n'.format(FLAGS.image_dir_medium))
    f.write('# image_dir_low: {}\n'.format(FLAGS.image_dir_low))
    f.write('# image_dir_low2: {}\n'.format(FLAGS.image_dir_low2))
    f.write('# slide_list_filename_train: {}\n'.format(FLAGS.slide_list_filename_train))
    f.write('# slide_list_filename_valid: {}\n'.format(FLAGS.slide_list_filename_valid))
    f.write('# slide_list_filename_test: {}\n'.format(FLAGS.slide_list_filename_test))
    f.write('# patch_size: {}\n'.format(FLAGS.patch_size))
    f.write('# num_classes: {}\n'.format(FLAGS.num_classes))
    f.write('# pretrained: {}\n'.format(FLAGS.pretrained))
    f.write('# batch_size: {}\n'.format(FLAGS.batch_size))
    f.write('# learning_rate: {}\n'.format(FLAGS.learning_rate))
    f.write('# weight_decay: {}\n'.format(FLAGS.weight_decay))
    f.write('# num_epochs: {}\n'.format(FLAGS.num_epochs))
    f.write('# save_interval: {}\n'.format(FLAGS.save_interval))
    f.write('# metrics_file: {}\n'.format(FLAGS.metrics_file))
    f.write('# metrics_loss_file: {}\n'.format(FLAGS.metrics_loss_file))
    f.write('# metrics_acc_file: {}\n'.format(FLAGS.metrics_acc_file))
    f.write('# metrics_cm_file: {}\n'.format(FLAGS.metrics_cm_file))
    f.write('# test_loss_file: {}\n'.format(FLAGS.test_loss_file))
    f.write('# test_acc_file: {}\n'.format(FLAGS.test_acc_file))
    f.write('# test_cm_file: {}\n'.format(FLAGS.test_cm_file))    
    f.write('# epoch\tlearning_rate\ttraining_loss_high\ttraining_loss_medium\ttraining_loss_low\ttraining_loss_low2\ttraining_loss_result\ttraining_loss_total \
            \tvalidation_loss_high\tvalidation_loss_medium\tvalidation_loss_low\tvalidation_loss_low2\tvalidation_loss_result\tvalidation_loss_total\n')
    
with open(FLAGS.metrics_acc_file, 'w') as f:        
    f.write('# epoch\tlearning_rate\ttraining_acc_high\ttraining_acc_medium\ttraining_acc_low\ttraining_acc_low2\ttraining_acc_result\ttraining_acc_total \
            \tvalidation_acc_high\tvalidation_acc_medium\tvalidation_acc_low\tvalidation_acc_low2\tvalidation_acc_result\tvalidation_acc_total\n')

                        
with open(FLAGS.metrics_cm_file, 'w') as f:
    f.write('# epoch\tlearning_rate \
            \ttraining_label_benign_predicted_benign_high\ttraining_label_benign_predicted_malignant_high\ttraining_label_malignant_predicted_benign_high\ttraining_label_malignant_predicted_malignant_high \
            \ttraining_label_benign_predicted_benign_medium\ttraining_label_benign_predicted_malignant_medium\ttraining_label_malignant_predicted_benign_medium\ttraining_label_malignant_predicted_malignant_medium \
            \ttraining_label_benign_predicted_benign_low\ttraining_label_benign_predicted_malignant_low\ttraining_label_malignant_predicted_benign_low\ttraining_label_malignant_predicted_malignant_low \
            \ttraining_label_benign_predicted_benign_low2\ttraining_label_benign_predicted_malignant_low2\ttraining_label_malignant_predicted_benign_low2\ttraining_label_malignant_predicted_malignant_low2 \
            \ttraining_label_benign_predicted_benign_result\ttraining_label_benign_predicted_malignant_result\ttraining_label_malignant_predicted_benign_result\ttraining_label_malignant_predicted_malignant_result \
            \ttraining_label_benign_predicted_benign_total\ttraining_label_benign_predicted_malignant_total\ttraining_label_malignant_predicted_benign_total\ttraining_label_malignant_predicted_malignant_total \
            \tvalidation_label_benign_predicted_benign_high\tvalidation_label_benign_predicted_malignant_high\tvalidation_label_malignant_predicted_benign_high\tvalidation_label_malignant_predicted_malignant_high \
            \tvalidation_label_benign_predicted_benign_medium\tvalidation_label_benign_predicted_malignant_medium\tvalidation_label_malignant_predicted_benign_medium\tvalidation_label_malignant_predicted_malignant_medium \
            \tvalidation_label_benign_predicted_benign_low\tvalidation_label_benign_predicted_malignant_low\tvalidation_label_malignant_predicted_benign_low\tvalidation_label_malignant_predicted_malignant_low \
            \tvalidation_label_benign_predicted_benign_low2\tvalidation_label_benign_predicted_malignant_low2\tvalidation_label_malignant_predicted_benign_low2\tvalidation_label_malignant_predicted_malignant_low2 \
            \tvalidation_label_benign_predicted_benign_result\tvalidation_label_benign_predicted_malignant_result\tvalidation_label_malignant_predicted_benign_result\tvalidation_label_malignant_predicted_malignant_result \
            \tvalidation_label_benign_predicted_benign_total\tvalidation_label_benign_predicted_malignant_total\tvalidation_label_malignant_predicted_benign_total\tvalidation_label_malignant_predicted_malignant_total\n')

total_steps = len(data_loader_train)
best_acc = 0.0
min_val_loss = 100.0
    
for epoch in range(FLAGS.num_epochs):
    
    print('#################### EPOCH - {} ####################'.format(epoch + 1))
    
    print('******************** training ********************')

    pbar = tqdm(total=len(data_loader_train))
        
    model.train()
        
    num_predictions = 0
    
    running_loss_high = 0.0
    running_loss_medium = 0.0
    running_loss_low = 0.0
    running_loss_low2 = 0.0
    running_loss_result = 0.0
    running_loss_total = 0.0
    
    running_correct_high = 0
    running_correct_medium = 0
    running_correct_low = 0
    running_correct_low2 = 0
    running_correct_result = 0
    running_correct_total = 0
    
    label_list = []
    predicted_list_high = []
    predicted_list_medium = []
    predicted_list_low = []
    predicted_list_low2 = []
    predicted_list_result = []
    predicted_list_total = []
        
    for i, (img_paths, img_high, img_medium, img_low, img_low2, label) in enumerate(data_loader_train):
        
        # print('high: {}'.format(img_high.shape))
        # print('medium: {}'.format(img_medium.shape))
        # print('low: {}'.format(img_low.shape))
        # print('low2: {}'.format(img_low2.shape))
        # print('label: {}'.format(label.shape))
                    
        img_high, img_medium, img_low, img_low2, label = img_high.to(device), img_medium.to(device), img_low.to(device), img_low2.to(device), label.to(device)
        output_high, output_medium, output_low, output_low2, output_result = model(img_high, img_medium, img_low, img_low2)
        output_total = output_high + output_medium + output_low + output_low2 + output_result    
            
        optimizer.zero_grad()
        loss_high = criterion(output_high, label)
        loss_medium = criterion(output_medium, label)
        loss_low = criterion(output_low, label)
        loss_low2 = criterion(output_low2, label)
        loss_result = criterion(output_result, label)
        loss_total = loss_high + loss_medium + loss_low + loss_low2 + loss_result
        loss_total.backward()
        optimizer.step()
                                   
        _, predicted_high = torch.max(output_high, 1)
        _, predicted_medium = torch.max(output_medium, 1)
        _, predicted_low = torch.max(output_low, 1)
        _, predicted_low2 = torch.max(output_low2, 1)
        _, predicted_result = torch.max(output_result, 1)
        _, predicted_total = torch.max(output_total, 1)
        
        correct_high = (predicted_high == label).sum().item()
        correct_medium = (predicted_medium == label).sum().item()
        correct_low = (predicted_low == label).sum().item()
        correct_low2 = (predicted_low2 == label).sum().item()
        correct_result = (predicted_result == label).sum().item()
        correct_total = (predicted_total == label).sum().item()
        
        num_predictions += label.size(0)
        
        running_loss_high += loss_high.item() * label.size(0)
        running_loss_medium += loss_medium.item() * label.size(0)
        running_loss_low += loss_low.item() * label.size(0)
        running_loss_low2 += loss_low2.item() * label.size(0)
        running_loss_result += loss_result.item() * label.size(0)
        running_loss_total += loss_total.item() * label.size(0)
        
        running_correct_high += correct_high
        running_correct_medium += correct_medium
        running_correct_low += correct_low
        running_correct_low2 += correct_low2
        running_correct_result += correct_result
        running_correct_total += correct_total
            
        label_list += list(label.cpu().numpy())
        predicted_list_high += list(predicted_high.cpu().numpy())
        predicted_list_medium += list(predicted_medium.cpu().numpy())
        predicted_list_low += list(predicted_low.cpu().numpy())
        predicted_list_low2 += list(predicted_low2.cpu().numpy())
        predicted_list_result += list(predicted_result.cpu().numpy())     
        predicted_list_total += list(predicted_total.cpu().numpy())
            
        pbar.update(1)
        
    pbar.close()

    train_loss_high = running_loss_high / num_predictions
    train_loss_medium = running_loss_medium / num_predictions
    train_loss_low = running_loss_low / num_predictions
    train_loss_low2 = running_loss_low2 / num_predictions
    train_loss_result = running_loss_result / num_predictions
    train_loss_total = running_loss_total / num_predictions
   
    train_acc_high = running_correct_high / num_predictions
    train_acc_medium = running_correct_medium / num_predictions
    train_acc_low = running_correct_low / num_predictions
    train_acc_low2 = running_correct_low2 / num_predictions
    train_acc_result = running_correct_result / num_predictions
    train_acc_total = running_correct_total / num_predictions
        
    print('Training loss high: {:.4f}\tTraining loss medium: {:.4f}\tTraining loss low: {:.4f}\tTraining loss low2: {:.4f}\tTraining loss result: {:.4f}\tTraining loss total: {:.4f}'.format(train_loss_high, train_loss_medium, train_loss_low, train_loss_low2, train_loss_result, train_loss_total))
    print('Training accuracy high: {:.4f}\tTraining accuracy medium: {:.4f}\tTraining accuracy low: {:.4f}\tTraining accuracy low2: {:.4f}\tTraining accuracy result: {:.4f}\tTraining accuracy total: {:.4f}'.format(train_acc_high, train_acc_medium, train_acc_low, train_acc_low2, train_acc_result, train_acc_total))

    # confusion matrix
    cm_train_high = confusion_matrix(label_list, predicted_list_high, labels=[0, 1])
    cm_train_medium = confusion_matrix(label_list, predicted_list_medium, labels=[0, 1])
    cm_train_low = confusion_matrix(label_list, predicted_list_low, labels=[0, 1])
    cm_train_low2 = confusion_matrix(label_list, predicted_list_low2, labels=[0, 1])
    cm_train_result = confusion_matrix(label_list, predicted_list_result, labels=[0, 1])
    cm_train_total = confusion_matrix(label_list, predicted_list_total, labels=[0, 1])
    
    
    print('******************** validation ********************')

    pbar2 = tqdm(total=len(data_loader_valid))

    # validation
    model.eval()
    
    num_predictions = 0
    
    running_loss_high = 0.0
    running_loss_medium = 0.0
    running_loss_low = 0.0
    running_loss_low2 = 0.0
    running_loss_result = 0.0
    running_loss_total = 0.0
    
    running_correct_high = 0
    running_correct_medium = 0
    running_correct_low = 0
    running_correct_low2 = 0
    running_correct_result = 0
    running_correct_total = 0
        
    label_list = []
    predicted_list_high = []
    predicted_list_medium = []
    predicted_list_low = []
    predicted_list_low2 = []
    predicted_list_result = []
    predicted_list_total = []
        
    with torch.no_grad():
        for i, (img_paths, img_high, img_medium, img_low, img_low2, label) in enumerate(data_loader_valid):
            
            # print('high: {}'.format(img_high.shape))
            # print('medium: {}'.format(img_medium.shape))
            # print('low: {}'.format(img_low.shape))
            # print('low2: {}'.format(img_low2.shape))
            # print('label: {}'.format(label.shape))
                        
            img_high, img_medium, img_low, img_low2, label = img_high.to(device), img_medium.to(device), img_low.to(device), img_low2.to(device), label.to(device)
            output_high, output_medium, output_low, output_low2, output_result = model(img_high, img_medium, img_low, img_low2)
            output_total = output_high + output_medium + output_low + output_low2 + output_result
            
            loss_high = criterion(output_high, label)
            loss_medium = criterion(output_medium, label)
            loss_low = criterion(output_low, label)
            loss_low2 = criterion(output_low2, label)
            loss_result = criterion(output_result, label)
            loss_total = loss_high + loss_medium + loss_low + loss_low2 + loss_result
            
            # print('loss_total: {}'.format(loss_total))
                        
            _, predicted_high = torch.max(output_high, 1)
            _, predicted_medium = torch.max(output_medium, 1)
            _, predicted_low = torch.max(output_low, 1)
            _, predicted_low2 = torch.max(output_low2, 1)
            _, predicted_result = torch.max(output_result, 1)
            _, predicted_total = torch.max(output_total, 1)
        
            correct_high = (predicted_high == label).sum().item()
            correct_medium = (predicted_medium == label).sum().item()
            correct_low = (predicted_low == label).sum().item()
            correct_low2 = (predicted_low2 == label).sum().item()
            correct_result = (predicted_result == label).sum().item()
            correct_total = (predicted_total == label).sum().item()

            num_predictions += label.size(0)
            
            running_loss_high += loss_high.item() * label.size(0)
            running_loss_medium += loss_medium.item() * label.size(0)
            running_loss_low += loss_low.item() * label.size(0)
            running_loss_low2 += loss_low2.item() * label.size(0)
            running_loss_result += loss_result.item() * label.size(0)
            running_loss_total += loss_total.item() * label.size(0)
        
            running_correct_high += correct_high
            running_correct_medium += correct_medium
            running_correct_low += correct_low
            running_correct_low2 += correct_low2
            running_correct_result += correct_result
            running_correct_total += correct_total
            
            label_list += list(label.cpu().numpy())
            predicted_list_high += list(predicted_high.cpu().numpy())
            predicted_list_medium += list(predicted_medium.cpu().numpy())
            predicted_list_low += list(predicted_low.cpu().numpy())
            predicted_list_low2 += list(predicted_low2.cpu().numpy())
            predicted_list_result += list(predicted_result.cpu().numpy())     
            predicted_list_total += list(predicted_total.cpu().numpy())
            
            pbar2.update(1)

    pbar2.close()
                
    valid_loss_high = running_loss_high / num_predictions
    valid_loss_medium = running_loss_medium / num_predictions
    valid_loss_low = running_loss_low / num_predictions
    valid_loss_low2 = running_loss_low2 / num_predictions
    valid_loss_result = running_loss_result / num_predictions
    valid_loss_total = running_loss_total / num_predictions
    
    valid_acc_high = running_correct_high / num_predictions
    valid_acc_medium = running_correct_medium / num_predictions
    valid_acc_low = running_correct_low / num_predictions
    valid_acc_low2 = running_correct_low2 / num_predictions
    valid_acc_result = running_correct_result / num_predictions
    valid_acc_total = running_correct_total / num_predictions

    # confusion matrix
    cm_valid_high = confusion_matrix(label_list, predicted_list_high, labels=[0, 1])
    cm_valid_medium = confusion_matrix(label_list, predicted_list_medium, labels=[0, 1])
    cm_valid_low = confusion_matrix(label_list, predicted_list_low, labels=[0, 1])
    cm_valid_low2 = confusion_matrix(label_list, predicted_list_low2, labels=[0, 1])
    cm_valid_result = confusion_matrix(label_list, predicted_list_result, labels=[0, 1])
    cm_valid_total = confusion_matrix(label_list, predicted_list_total, labels=[0, 1])
        
    # print('Epoch : {:d}'.format(epoch + 1))
    print('Validation loss high: {:.4f}\tValidation loss medium: {:.4f}\tValidation loss low: {:.4f}\tValidation loss low2: {:.4f}\tValidation loss result: {:.4f}\tValidation loss total: {:.4f}' \
          .format(valid_loss_high, valid_loss_medium, valid_loss_low, valid_loss_low2, valid_loss_result, valid_loss_total))
    print('Validation accuracy high: {:.4f}\tValidation accuracy medium: {:.4f}\tValidation accuracy low: {:.4f}\tValidation accuracy low2: {:.4f}\tValidation accuracy result: {:.4f}\tValidation accuracy total: {:.4f}' \
          .format(valid_acc_high, valid_acc_medium, valid_acc_low, valid_acc_low2, valid_acc_result, valid_acc_total))
    # print('\n')
    

    
    with open(FLAGS.metrics_loss_file, 'a') as f:
        f.write('{:d}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n' \
                .format(epoch + 1, optimizer.param_groups[0]['lr'], 
                        train_loss_high, train_loss_medium, train_loss_low, train_loss_low2, train_loss_result, train_loss_total, 
                        valid_loss_high, valid_loss_medium, valid_loss_low, valid_loss_low2, valid_loss_result, valid_loss_total))

    with open(FLAGS.metrics_acc_file, 'a') as f:
        f.write('{:d}\t{:.8f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n' \
                .format(epoch + 1, optimizer.param_groups[0]['lr'], 
                        train_acc_high, train_acc_medium, train_acc_low, train_acc_low2, train_acc_result, train_acc_total, 
                        valid_acc_high, valid_acc_medium, valid_acc_low, valid_acc_low2, valid_acc_result, valid_acc_total))

    with open(FLAGS.metrics_cm_file, 'a') as f:
        f.write('{:d}\t{:.8f}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\n' \
                .format(epoch + 1, optimizer.param_groups[0]['lr'], 
                        cm_train_high[0, 0], cm_train_high[0, 1], cm_train_high[1, 0], cm_train_high[1, 1],
                        cm_train_medium[0, 0], cm_train_medium[0, 1], cm_train_medium[1, 0], cm_train_medium[1, 1],
                        cm_train_low[0, 0], cm_train_low[0, 1], cm_train_low[1, 0], cm_train_low[1, 1],
                        cm_train_low2[0, 0], cm_train_low2[0, 1], cm_train_low2[1, 0], cm_train_low2[1, 1],
                        cm_train_result[0, 0], cm_train_result[0, 1], cm_train_result[1, 0], cm_train_result[1, 1],
                        cm_train_total[0, 0], cm_train_total[0, 1], cm_train_total[1, 0], cm_train_total[1, 1], 
                        cm_valid_high[0, 0], cm_valid_high[0, 1], cm_valid_high[1, 0], cm_valid_high[1, 1],
                        cm_valid_medium[0, 0], cm_valid_medium[0, 1], cm_valid_medium[1, 0], cm_valid_medium[1, 1],
                        cm_valid_low[0, 0], cm_valid_low[0, 1], cm_valid_low[1, 0], cm_valid_low[1, 1],
                        cm_valid_low2[0, 0], cm_valid_low2[0, 1], cm_valid_low2[1, 0], cm_valid_low2[1, 1],
                        cm_valid_result[0, 0], cm_valid_result[0, 1], cm_valid_result[1, 0], cm_valid_result[1, 1],
                        cm_valid_total[0, 0], cm_valid_total[0, 1], cm_valid_total[1, 0], cm_valid_total[1, 1]))


    if (valid_loss_result < min_val_loss) or ((epoch + 1) % FLAGS.save_interval == 0):
        model_weights_filename = FLAGS.model_dir + 'model_weights' + current_time + '__' + str(epoch + 1) + '.pth'
        state_dict = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state_dict, model_weights_filename)
        print('Model weights saved in file: {}'.format(model_weights_filename))
           
    if valid_loss_result < min_val_loss:
        min_val_loss = valid_loss_result


##################################################################################################################################

print('******************** testing ********************')

pbar = tqdm(total=len(data_loader_test))

model.eval()
        
num_predictions = 0

running_loss_high = 0.0
running_loss_medium = 0.0
running_loss_low = 0.0
running_loss_low2 = 0.0
running_loss_result = 0.0
running_loss_total = 0.0
    
running_correct_high = 0
running_correct_medium = 0
running_correct_low = 0
running_correct_low2 = 0
running_correct_result = 0
running_correct_total = 0
        
label_list = []
predicted_list_high = []
predicted_list_medium = []
predicted_list_low = []
predicted_list_low2 = []
predicted_list_result = []
predicted_list_total = []
        
with torch.no_grad():
    for i, (img_paths, img_high, img_medium, img_low, img_low2, label) in enumerate(data_loader_test):
                
        img_high, img_medium, img_low, img_low2, label = img_high.to(device), img_medium.to(device), img_low.to(device), img_low2.to(device), label.to(device)
        output_high, output_medium, output_low, output_low2, output_result = model(img_high, img_medium, img_low, img_low2)
        output_total = output_high + output_medium + output_low + output_low2 + output_result
            
        loss_high = criterion(output_high, label)
        loss_medium = criterion(output_medium, label)
        loss_low = criterion(output_low, label)
        loss_low2 = criterion(output_low2, label)
        loss_result = criterion(output_result, label)
        loss_total = loss_high + loss_medium + loss_low + loss_low2 + loss_result
                        
        _, predicted_high = torch.max(output_high, 1)
        _, predicted_medium = torch.max(output_medium, 1)
        _, predicted_low = torch.max(output_low, 1)
        _, predicted_low2 = torch.max(output_low2, 1)
        _, predicted_result = torch.max(output_result, 1)
        _, predicted_total = torch.max(output_total, 1)
        
        correct_high = (predicted_high == label).sum().item()
        correct_medium = (predicted_medium == label).sum().item()
        correct_low = (predicted_low == label).sum().item()
        correct_low2 = (predicted_low2 == label).sum().item()
        correct_result = (predicted_result == label).sum().item()
        correct_total = (predicted_total == label).sum().item()
        
        running_loss_high += loss_high.item() * label.size(0)
        running_loss_medium += loss_medium.item() * label.size(0)
        running_loss_low += loss_low.item() * label.size(0)
        running_loss_low2 += loss_low2.item() * label.size(0)
        running_loss_result += loss_result.item() * label.size(0)
        running_loss_total += loss_total.item() * label.size(0)
        
        num_predictions += label.size(0)
        running_correct_high += correct_high
        running_correct_medium += correct_medium
        running_correct_low += correct_low
        running_correct_low2 += correct_low2
        running_correct_result += correct_result
        running_correct_total += correct_total
            
        label_list += list(label.cpu().numpy())
        predicted_list_high += list(predicted_high.cpu().numpy())
        predicted_list_medium += list(predicted_medium.cpu().numpy())
        predicted_list_low += list(predicted_low.cpu().numpy())
        predicted_list_low2 += list(predicted_low2.cpu().numpy())
        predicted_list_result += list(predicted_result.cpu().numpy())     
        predicted_list_total += list(predicted_total.cpu().numpy())
        
        pbar.update(1)
                
test_loss_high = running_loss_high / num_predictions
test_loss_medium = running_loss_medium / num_predictions
test_loss_low = running_loss_low / num_predictions
test_loss_low2 = running_loss_low2 / num_predictions
test_loss_result = running_loss_result / num_predictions
test_loss_total = running_loss_total / num_predictions
    
test_acc_high = running_correct_high / num_predictions
test_acc_medium = running_correct_medium / num_predictions
test_acc_low = running_correct_low / num_predictions
test_acc_low2 = running_correct_low2 / num_predictions
test_acc_result = running_correct_result / num_predictions
test_acc_total = running_correct_total / num_predictions
        
# confusion matrix
cm_test_high = confusion_matrix(label_list, predicted_list_high, labels=[0, 1])
cm_test_medium = confusion_matrix(label_list, predicted_list_medium, labels=[0, 1])
cm_test_low = confusion_matrix(label_list, predicted_list_low, labels=[0, 1])
cm_test_low2 = confusion_matrix(label_list, predicted_list_low2, labels=[0, 1])
cm_test_result = confusion_matrix(label_list, predicted_list_result, labels=[0, 1])
cm_test_total = confusion_matrix(label_list, predicted_list_total, labels=[0, 1])
        
pbar.close()

with open(FLAGS.test_loss_file, 'w') as f:
    f.write('# test_loss_high\ttest_loss_medium\ttest_loss_low\ttest_loss_low2\ttest_loss_result\ttest_loss_total\n')
    f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(test_loss_high, test_loss_medium, test_loss_low, test_loss_low2, test_loss_result, test_loss_total))

with open(FLAGS.test_acc_file, 'w') as f:
    f.write('# test_acc_high\ttest_acc_medium\ttest_acc_low\ttest_acc_low2\ttest_acc_result\ttest_acc_total\n')
    f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(test_acc_high, test_acc_medium, test_acc_low, test_acc_low2, test_acc_result, test_acc_total))

with open(FLAGS.test_cm_file, 'w') as f:
    f.write('# test_label_benign_predicted_benign_high\ttest_label_benign_predicted_malignant_high\ttest_label_malignant_predicted_benign_high\ttest_label_malignant_predicted_malignant_high \
            \ttest_label_benign_predicted_benign_medium\ttest_label_benign_predicted_malignant_medium\ttest_label_malignant_predicted_benign_medium\ttest_label_malignant_predicted_malignant_medium \
            \ttest_label_benign_predicted_benign_low\ttest_label_benign_predicted_malignant_low\ttest_label_malignant_predicted_benign_low\ttest_label_malignant_predicted_malignant_low \
            \ttest_label_benign_predicted_benign_low2\ttest_label_benign_predicted_malignant_low2\ttest_label_malignant_predicted_benign_low2\ttest_label_malignant_predicted_malignant_low2 \
            \ttest_label_benign_predicted_benign_result\ttest_label_benign_predicted_malignant_result\ttest_label_malignant_predicted_benign_result\ttest_label_malignant_predicted_malignant_result\t \
            \ttest_label_benign_predicted_benign_total\ttest_label_benign_predicted_malignant_total\ttest_label_malignant_predicted_benign_total\ttest_label_malignant_predicted_malignant_total\n')

    f.write('{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}\n' \
            .format(cm_test_high[0, 0], cm_test_high[0, 1], cm_test_high[1, 0], cm_test_high[1, 1],
                    cm_test_medium[0, 0], cm_test_medium[0, 1], cm_test_medium[1, 0], cm_test_medium[1, 1],
                    cm_test_low[0, 0], cm_test_low[0, 1], cm_test_low[1, 0], cm_test_low[1, 1],
                    cm_test_low2[0, 0], cm_test_low2[0, 1], cm_test_low2[1, 0], cm_test_low2[1, 1],
                    cm_test_result[0, 0], cm_test_result[0, 1], cm_test_result[1, 0], cm_test_result[1, 1],
                    cm_test_total[0, 0], cm_test_total[0, 1], cm_test_total[1, 0], cm_test_total[1, 1]))


