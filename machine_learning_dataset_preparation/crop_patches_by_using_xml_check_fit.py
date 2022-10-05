import os
import sys
import xml.etree.ElementTree as ET
import openslide
import matplotlib.colors as mplt_color
import matplotlib.pyplot as plt
import numpy as np
import argparse
import fnmatch
from PIL import Image
import scipy.misc
import imageio
from datetime import datetime
import cv2
from tqdm import tqdm


annotation_group_dict = {'p3': 'pattern3',
                         'P3': 'pattern3',
                         'p4': 'pattern4',
                         'P4': 'pattern4',
                         'b': 'benign',
                         'B': 'benign',
                         'tangential_benign': 'tangential_benign',
                         'tangential_malignant': 'tangential_malignant',
                         'pattern3': 'pattern3',
                         'pattern4': 'pattern4',
                         'benign': 'benign',
                         'unknown': 'unknown',
                         'PIN': 'PIN',
                         'artefact': 'artefact',
                         'artifact': 'artefact',
                         'Artefact': 'artefact',
                         't': 'tangential_benign',
                         'tangential': 'tangential_benign'}

annotation_label_dict = {'pattern3': 1,
                         'pattern4': 2,
                         'benign': 3,
                         'tangential_benign': 4,
                         'tangential_malignant': 5,
                         'unknown': 6,
                         'PIN': 7,
                         'artefact': 8}

annotation_group_color_dict = {'pattern3': [255,0,0],
                               'pattern4': [0,255,0],
                               'benign': [0,0,255],
                               'tangential_benign': [255,255,0],
                               'tangential_malignant': [0,255,255],
                               'unknown': [0,128,0],
                               'PIN': [0,128,128],
                               'artefact': [128,128,128]}


def check_fit_into_frame(ref_annotation_dict=None, candidate_annotation_dict=None, im_read_level=0, im_read_size=None):
    """
    Checks if a candidate annotation fits into the frame defined by the reference annotation.

    Args:
        ref_annotation_dict (dict): Reference annotation dictionary defining the frame.
        candidate_annotation_dict (dict): Candidate annotation dictionary to be checked if it fits into the frame.
        im_read_level (int): WSI level to read images from. Default: 0
        im_read_size (tuple): Image size at the im_read_level. Note that images are assumed to be square in this implementation.
  
    Returns:
        flag (bool): Returns true if candidate annotation fits into the reference's frame, false otherwise.

    """


    if im_read_level != 0:
        print('ERROR: im_read_level != 0')
        sys.exit()

    X_cm_ref = ref_annotation_dict['X_cm']
    Y_cm_ref = ref_annotation_dict['Y_cm']

    X_min_frame = X_cm_ref - int(im_read_size[0]/2)
    Y_min_frame = Y_cm_ref - int(im_read_size[0]/2)
    X_max_frame = X_min_frame + im_read_size[0]
    Y_max_frame = Y_min_frame + im_read_size[0]

    X_min_candidate = candidate_annotation_dict['X_min']
    X_max_candidate = candidate_annotation_dict['X_max']
    Y_min_candidate = candidate_annotation_dict['Y_min']
    Y_max_candidate = candidate_annotation_dict['Y_max']
    

    # check the size of candidate annotation
    if (X_min_candidate < X_min_frame) or (X_max_candidate > X_max_frame) or (Y_min_candidate < Y_min_frame) or (Y_max_candidate > Y_max_frame):
        return False
    else:
        return True

def get_img(slide=None, X_cm=0, Y_cm=0, im_read_level=0, im_read_size=None, res_ratio_target_to_read=1):
    """
    Reads an image from the given slide at specified location. Location is specified by the coordinates 
    of the patch's center of mass. Slide level to read from and image size at this level 
    are provided as input. If the final image level is different than the read level, image is scaled 
    before returning.

    Args:
        slide (obj): Slide object
        X_cm (int): X coordinate of the patch's center of mass. Default: 0
        Y_cm (int): Y coordinate of the patch's center of mass. Default: 0
        im_read_level (int): WSI level to read images from. Default: 0
        im_read_size (tuple): Image size at the im_read_level. Note that images are assumed to be square in this implementation.
        res_ratio_target_to_read (float): If the final image level is different than the read level. Default: 1
  
    Returns:
        im_arr (ndarray): Returns image array

    """

    if im_read_level != 0:
        print('ERROR: im_read_level != 0')
        sys.exit()

    X_min_frame = X_cm - int(im_read_size[0]/2)
    Y_min_frame = Y_cm - int(im_read_size[0]/2)

    im = slide.read_region((X_min_frame, Y_min_frame), im_read_level, im_read_size)
    im_arr = np.array(im)[:,:,0:3]

    if res_ratio_target_to_read>1:
        im_arr = np.array(im.resize((patch_size,patch_size),Image.ANTIALIAS))[:,:,0:3]

    # print(im_read_size)

    return im_arr

def get_mask_complete_glands(annotations_dict_list = None, center_annotation_dict = None, im_read_level = 0, im_read_size = None, res_ratio_target_to_read=1):
    """
    Create a mask by checking a list of candidate annotations if they fit into a frame 
    defined by a reference annotation (annotation of the gland at the center of the patch). 
    Mask will include only the annotations fitting into the frame completely.

    Args:
        annotations_dict_list (list): List of candidate annotations to be checked if they fit into the frame.
        center_annotation_dict (dict): Reference annotation dictionary defining the frame.
        im_read_level (int): WSI level that images are read from. Default: 0
        im_read_size (tuple): Image size at the im_read_level. Note that images are assumed to be square in this implementation.
        res_ratio_target_to_read (float): If the final image level is different than the read level. Default: 1
  
    Returns:
        label_list (list): List of labels of annotations fitting into the frame
        canvas_instance (ndarray): 2D instances mask array 
        canvas_binary_all (list): List of binary masks of annotations within the frame
        canvas_color (ndarray): 3D color-coded instances mask array
        center_id (int): Id of the center annotation
        center_checked (int): Flag indicating if center annotation is checked by the pathologist. 1: checked, 0: not checked
        error_flag (bool): Error flag. True if there is an error, false otherwise.

    """

    if im_read_level != 0:
        print('ERROR: im_read_level != 0')
        sys.exit()

    res_ratio_read_to_target = 1.0/res_ratio_target_to_read
    # print('res ratio read to target: {}'.format(res_ratio_read_to_target))
    mask_size = int(im_read_size[0]*res_ratio_read_to_target)

    X_cm_center = center_annotation_dict['X_cm']
    Y_cm_center = center_annotation_dict['Y_cm']

    X_offset = int(X_cm_center*res_ratio_read_to_target) - int(mask_size/2.0)
    Y_offset = int(Y_cm_center*res_ratio_read_to_target) - int(mask_size/2.0)

    num_instances = 1
    canvas_instance = np.zeros((mask_size,mask_size), dtype=np.uint8)
    canvas_color = np.zeros((mask_size,mask_size,3), dtype=np.uint8)
    canvas_binary_all = []
    label_list = list()
    center_id = 0
    center_checked = 0
    error_flag = False
    
    for i in range(len(annotations_dict_list)):
        temp_annotation_dict = annotations_dict_list[i]
        
        center_flag = (temp_annotation_dict['X_cm'] == center_annotation_dict['X_cm']) and (temp_annotation_dict['Y_cm'] == center_annotation_dict['Y_cm'])    


        fit_into_frame_flag = check_fit_into_frame(    ref_annotation_dict=center_annotation_dict, 
                                                    candidate_annotation_dict=temp_annotation_dict,
                                                    im_read_level=im_read_level,
                                                    im_read_size=im_read_size)
        # exclude partial non-center glands
        if not fit_into_frame_flag and not center_flag:
            continue
        
        canvas_binary = np.zeros((mask_size,mask_size), dtype=np.uint8)            
        X_arr = np.asarray(temp_annotation_dict['X_arr']*res_ratio_read_to_target, dtype=int) - X_offset
        Y_arr = np.asarray(temp_annotation_dict['Y_arr']*res_ratio_read_to_target, dtype=int) - Y_offset
        annotation_group = temp_annotation_dict['annotation_group']

        pts = np.hstack((X_arr[:,np.newaxis],Y_arr[:,np.newaxis]))
        cv2.drawContours(canvas_binary, [pts], 0, (1), -1)
        # annotation of center gland and patch must overlap
        if np.sum(canvas_binary) == 0:
            error_flag = True
        cv2.drawContours(canvas_instance, [pts], 0, (num_instances), -1)
        
        canvas_binary_all.append(canvas_binary)
        rgb_color = annotation_group_color_dict[annotation_group]
        cv2.drawContours(canvas_color, [pts], 0, rgb_color, -1)
        label_list.append(annotation_label_dict[annotation_group])            
        # check if current annotation is the annotation of the center gland
        if center_flag:
            center_id = num_instances - 1                        
            if 'checked' in temp_annotation_dict['name']:
                center_checked = 1
        num_instances += 1
    
    # print(len(canvas_binary_all))
    if len(canvas_binary_all) == 1:
        canvas_binary_all = canvas_binary_all[0][np.newaxis, :]
    elif len(canvas_binary_all) > 1:
        canvas_binary_all = np.stack(canvas_binary_all)
        
    # print('annotation name: {}\tnum instances: {}\tlabel list: {}'.format(center_annotation_dict['name'], num_instances, len(label_list)))
    # print('num_instances: {}'.format(num_instances))
    # print('len_labels: {}'.format(len(label_list)))
    # print('num_masks: {}'.format(canvas_binary_all.shape[0]))
    
    assert len(label_list) == canvas_binary_all.shape[0]
    return label_list, canvas_instance, canvas_binary_all, canvas_color, center_id, center_checked, error_flag
        

# include partial glands at edges of frame without resizing mask
def get_mask_complete_and_partial_glands(annotations_dict_list = None, center_annotation_dict = None, im_read_level = 0, im_read_size = None, res_ratio_target_to_read=1):
    """
    Create a mask by checking a list of candidate annotations if they fit into a frame 
    defined by a reference annotation (annotation of the gland at the center of the patch). 
    Mask will include all the annotations falling into the frame (both partial and complete).

    Args:
        annotations_dict_list (list): List of candidate annotations to be checked if they fit into the frame.
        center_annotation_dict (dict): Reference annotation dictionary defining the frame.
        im_read_level (int): WSI level that images are read from. Default: 0
        im_read_size (tuple): Image size at the im_read_level. Note that images are assumed to be square in this implementation.
        res_ratio_target_to_read (float): If the final image level is different than the read level. Default: 1
  
    Returns:
        label_list (list): List of labels of annotations fitting into the frame
        canvas_instance (ndarray): 2D instances mask array 
        canvas_binary_all (list): List of binary masks of annotations within the frame
        canvas_color (ndarray): 3D color-coded instances mask array
        center_id (int): Id of the center annotation
        center_checked (int): Flag indicating if center annotation is checked by the pathologist. 1: checked, 0: not checked
        error_flag (bool): Error flag. True if there is an error, false otherwise.

    """

    if im_read_level != 0:
        print('ERROR: im_read_level != 0')
        sys.exit()

    res_ratio_read_to_target = 1.0/res_ratio_target_to_read
    mask_size = patch_size
    # print('mask_size: {}'.format(mask_size))

    # shrink image
    X_cm_center = int(center_annotation_dict['X_cm']*res_ratio_read_to_target)
    Y_cm_center = int(center_annotation_dict['Y_cm']*res_ratio_read_to_target)

    X_offset = X_cm_center - int(mask_size/2.0)
    Y_offset = Y_cm_center - int(mask_size/2.0)

    num_instances = 1
    canvas_instance = np.zeros((mask_size,mask_size), dtype=np.uint8)
    canvas_color = np.zeros((mask_size,mask_size,3), dtype=np.uint8)
    canvas_binary_all = []
    label_list = list()
    center_id = 0
    center_checked = 0
    error_flag = False
    
    for i in range(len(annotations_dict_list)):
        temp_annotation_dict = annotations_dict_list[i]
        canvas_binary = np.zeros((mask_size,mask_size), dtype=np.uint8)    
        d_X = np.asarray(temp_annotation_dict['X_arr']*res_ratio_read_to_target, dtype=int) - X_offset
        d_Y = np.asarray(temp_annotation_dict['Y_arr']*res_ratio_read_to_target, dtype=int) - Y_offset
        annotation_group = temp_annotation_dict['annotation_group']
        # check whether there are any contour points inside the frame (points at the edge of the frame are excluded) 
        in_X = (d_X > 0) & (d_X < (mask_size - 1))
        in_Y = (d_Y > 0) & (d_Y < (mask_size - 1))
        in_sum = np.sum(in_X & in_Y)
        in_flag = in_sum > 0      
        center_flag = (temp_annotation_dict['X_cm'] == center_annotation_dict['X_cm']) and (temp_annotation_dict['Y_cm'] == center_annotation_dict['Y_cm'])    
        if center_flag or in_flag:                    
            # print('contour point lies in frame')
            pts = np.hstack((d_X[:,np.newaxis], d_Y[:,np.newaxis]))
            # if get_area(d_X[in_X & in_Y], d_Y[in_X & in_Y]) == 0:                
            #    print(d_X[in_X & in_Y], d_Y[in_X & in_Y])
            #    continue
            cv2.drawContours(canvas_binary, [pts], 0, (1), -1)

            # annotation of center gland and patch must overlap
            if np.sum(canvas_binary) == 0:
                error_flag = True
            cv2.drawContours(canvas_instance, [pts], 0, (num_instances), -1)
            
            canvas_binary_all.append(canvas_binary)
            rgb_color = annotation_group_color_dict[annotation_group]
            cv2.drawContours(canvas_color, [pts], 0, rgb_color, -1)
            label_list.append(annotation_label_dict[annotation_group])            
            # check if current annotation is the annotation of the center gland
            if center_flag:
                center_id = num_instances - 1                        
                if 'checked' in temp_annotation_dict['name']:
                    center_checked = 1
            num_instances += 1
    
    if len(canvas_binary_all) == 1:
        canvas_binary_all = canvas_binary_all[0][np.newaxis, :]
    elif len(canvas_binary_all) > 1:
        canvas_binary_all = np.stack(canvas_binary_all)
    # print('annotation name: {}\tnum instances: {}\tlabel list: {}'.format(center_annotation_dict['name'], num_instances, len(label_list)))
    # print('num_instances: {}'.format(num_instances))
    # print('len_labels: {}'.format(len(label_list)))
    # print('num_masks: {}'.format(canvas_binary_all.shape[0]))
    
    assert len(label_list) == canvas_binary_all.shape[0]
    return label_list, canvas_instance, canvas_binary_all, canvas_color, center_id, center_checked, error_flag

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--slide_dir', type=str, default='../WSIs', help='Directory containing whole slide images and xml files of manual annotations')
parser.add_argument('--slide_list_filename', type=str, default='slide_ids_list.txt', help='List of slide ids in dataset')
parser.add_argument('--check_fit_patch_level', type=int, default=1, help='Defines resolution at which to check whether centre gland fits into frame where 0 represents 40X and 1 represents 20X etc')
parser.add_argument('--patch_level', type=int, default=1, help='Defines resolution of cropped patches where 0 represents 40X and 1 represents 20X etc')
parser.add_argument('--patch_size', type=int, default=512, help='Patch size')
parser.add_argument('--include_peripheral_partial_glands', type=str2bool, default=True, help='Include partial glands at edges of the patch?')
parser.add_argument('--out_dir', type=str, default='../Images/gland_classification', help='Directory where cropped patches with centred glands are saved to')

FLAGS = parser.parse_args()

if not os.path.exists("logs"):
    try:
        os.makedirs("logs")
    except:
        print("An exception occurred!")

current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")

slide_list_filename = FLAGS.slide_list_filename
slide_dir = FLAGS.slide_dir
patch_level = FLAGS.patch_level
check_fit_patch_level = FLAGS.check_fit_patch_level
patch_size = FLAGS.patch_size
target_level_res = 0.25*(2**patch_level)
check_fit_level_res = 0.25*(2**check_fit_patch_level)
include_peripheral_partial_glands = FLAGS.include_peripheral_partial_glands

if include_peripheral_partial_glands:
    out_dir = FLAGS.out_dir + '/cropped_patches__complete_and_partial_glands__' + str(int(check_fit_level_res*100)) + '__' + str(int(target_level_res*100)) + '__' + str(patch_size)
else:
    out_dir = FLAGS.out_dir + '/cropped_patches__complete_glands__' + str(int(check_fit_level_res*100)) + '__' + str(int(target_level_res*100)) + '__' + str(patch_size)
    
print(out_dir)
if not os.path.exists(out_dir):
    try:
        os.makedirs(out_dir)
    except:
        print("An exception occurred!")

cropped_patches_info = out_dir + '/cropped_patches_info2' + current_time + '.txt'

# print(cropped_patches_info)
if os.path.isfile(cropped_patches_info):
    os.remove(cropped_patches_info)

with open(cropped_patches_info, 'a') as f_cropped_patches_info:
    f_cropped_patches_info.write('# wsi_id\tnumber_of_patches\n')

# get slide ids
slide_id_arr = np.loadtxt(slide_list_filename, dtype=str, comments='#', delimiter='\t')
num_slides = 1#slide_id_arr.shape[0]
# print('slide_id_arr:{}'.format(slide_id_arr))
# print('slide_id_arr.shape:{}'.format(slide_id_arr.shape))


for i in range(num_slides):
    slide_id = slide_id_arr[i]
    patient_id = slide_id[:11]

    slide_path = os.path.join(slide_dir,patient_id,slide_id + '.svs')
    # print(slide_path)

    slide = openslide.OpenSlide(slide_path)

    # print(slide.level_dimensions)
    val_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
    # print('Level0 Resolution:%3.2f' % val_x)

    if val_x < 0.3: # resolution:0.25um/pixel
        current_res = 0.25
    elif val_x < 0.6: # resolution:0.5um/pixel
        current_res = 0.5

    im_read_level = 0
    read_level_res = current_res
    res_ratio_target_to_read = target_level_res / read_level_res
    # print('res_ratio_target_to_read:{}'.format(res_ratio_target_to_read))
    im_read_size = (int(patch_size*res_ratio_target_to_read), int(patch_size*res_ratio_target_to_read))
    # print('im_read_size: {}'.format(im_read_size))
    
    res_ratio_fit_patch_level_to_read = check_fit_level_res / read_level_res
    im_read_size_fit_patch_level = (int(patch_size*res_ratio_fit_patch_level_to_read), int(patch_size*res_ratio_fit_patch_level_to_read))

    xml_file_path = os.path.join(slide_dir,patient_id,slide_id + '.xml')
    # print(xml_file_path)

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    Annotations = root[0]
    AnnotationGroups = root[1]

    # read all annotations
    annotations_dict_list = list()
    for Annotation  in Annotations:
        X_list = list()
        Y_list = list()

        Annotation_PartOfGroup = Annotation.attrib['PartOfGroup']
        annotation_name = Annotation.attrib['Name']
        if Annotation_PartOfGroup not in annotation_group_dict:
            continue

        annotation_group = annotation_group_dict[Annotation_PartOfGroup]

        Coordinates = Annotation[0]
        for Coordinate in Coordinates:
            X_list.append(int(float(Coordinate.attrib['X'])))
            Y_list.append(int(float(Coordinate.attrib['Y'])))

        X_list.append(X_list[0])
        Y_list.append(Y_list[0])

        X_arr = np.array(X_list)
        Y_arr = np.array(Y_list)

        # find min and max
        X_min = np.amin(X_arr)
        X_max = np.amax(X_arr)
        Y_min = np.amin(Y_arr)
        Y_max = np.amax(Y_arr)

        # calculate center of mass
        X_cm = int(np.mean(X_arr))
        Y_cm = int(np.mean(Y_arr))
        # print('X_cm={}, Y_cm={}'.format(X_cm, Y_cm))

        temp_annotation_dict = dict()
        temp_annotation_dict['annotation_group'] = annotation_group
        temp_annotation_dict['X_arr'] = X_arr
        temp_annotation_dict['Y_arr'] = Y_arr
        temp_annotation_dict['X_min'] = X_min
        temp_annotation_dict['X_max'] = X_max
        temp_annotation_dict['Y_min'] = Y_min
        temp_annotation_dict['Y_max'] = Y_max
        temp_annotation_dict['X_cm'] = X_cm
        temp_annotation_dict['Y_cm'] = Y_cm
        temp_annotation_dict['name'] = annotation_name

        annotations_dict_list.append(temp_annotation_dict)

    outdir = out_dir + '/' + slide_id
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cropped_patches_filelist = outdir + '/cropped_patches_filelist.txt'
    # print(cropped_patches_filelist)
    if os.path.isfile(cropped_patches_filelist):
        os.remove(cropped_patches_filelist)
        
    with open(cropped_patches_filelist, 'a') as f_cropped_patches_filelist:
        f_cropped_patches_filelist.write('#patch_id\tX_cm\tY_cm\tcenter_id\tcenter_checked\tcenter_label\n')
        

    outdir_img = outdir + '/img'
    if not os.path.exists(outdir_img):
        os.makedirs(outdir_img)

    outdir_label = outdir + '/label'
    if not os.path.exists(outdir_label):
        os.makedirs(outdir_label)

    outdir_mask = outdir + '/mask'
    if not os.path.exists(outdir_mask):
        os.makedirs(outdir_mask)
        
    outdir_binary_masks = outdir + '/binary_masks'
    if not os.path.exists(outdir_binary_masks):
        os.makedirs(outdir_binary_masks)

    outdir_mask_color = outdir + '/mask_color'
    if not os.path.exists(outdir_mask_color):
        os.makedirs(outdir_mask_color)
    
    
    num_cropped_patches = 0
    large_annotation_count = 0
    pbar = tqdm(total=len(annotations_dict_list))
    
    # skipped = []
    
    # process all annotations one-by-one as a central annotation
    for i in range(len(annotations_dict_list)):
        temp_annotation_dict = annotations_dict_list[i]
        
        fit_into_frame_flag = check_fit_into_frame( ref_annotation_dict = temp_annotation_dict,
                                                    candidate_annotation_dict = temp_annotation_dict,
                                                    im_read_level = im_read_level,
                                                    im_read_size = im_read_size_fit_patch_level)
        # skip annotations not fitting into frame
        if not fit_into_frame_flag:
            large_annotation_count += 1
            # print(temp_annotation_dict['name'])
            # skipped.append(Annotations[i].attrib['Name'])

            pbar.update(1)

            continue

        temp_img = get_img(    slide=slide, 
                            X_cm=temp_annotation_dict['X_cm'],
                            Y_cm=temp_annotation_dict['Y_cm'],
                            im_read_level = im_read_level,
                            im_read_size = im_read_size,
                            res_ratio_target_to_read = res_ratio_target_to_read)
        
        
        if include_peripheral_partial_glands:            
            temp_label_list, temp_mask, temp_binary_masks, temp_mask_color, temp_center_id, temp_center_checked, temp_error_flag = get_mask_complete_and_partial_glands(annotations_dict_list = annotations_dict_list,
                                                                                                                                                                        center_annotation_dict = temp_annotation_dict,
                                                                                                                                                                        im_read_level = im_read_level,
                                                                                                                                                                        im_read_size = im_read_size,
                                                                                                                                                                        res_ratio_target_to_read = res_ratio_target_to_read)
        else:
            temp_label_list, temp_mask, temp_binary_masks, temp_mask_color, temp_center_id, temp_center_checked, temp_error_flag = get_mask_complete_glands(annotations_dict_list = annotations_dict_list,
                                                                                                                                                            center_annotation_dict = temp_annotation_dict,
                                                                                                                                                            im_read_level = im_read_level,
                                                                                                                                                            im_read_size = im_read_size,
                                                                                                                                                            res_ratio_target_to_read = res_ratio_target_to_read)
        if temp_error_flag:
            print('error flag', outdir_img + '/' + str(num_cropped_patches) + '.png')

            pbar.update(1)
            
            continue
        
        outfile_img = outdir_img + '/' + str(num_cropped_patches) + '.png'
        # print(outfile_img)
        # scipy.misc.imsave(outfile_img, temp_img)
        imageio.imwrite(outfile_img, temp_img)

        outfile_label = outdir_label + '/' + str(num_cropped_patches) + '.txt'
        # print(outfile_label)
        np.savetxt(outfile_label, np.asarray(temp_label_list, dtype=np.uint8), comments='#', delimiter='\t', fmt='%d')

        outfile_mask = outdir_mask + '/' + str(num_cropped_patches) + '_mask.png'
        # print(outfile_mask)
        # scipy.misc.imsave(outfile_mask, temp_mask)
        imageio.imwrite(outfile_mask, temp_mask)
        
        # outfile_binary_masks = outdir_binary_masks + '/' + str(num_cropped_patches) + '_binary_masks'
        # np.save(outfile_binary_masks, temp_binary_masks)        
        
        # save binary masks as png files instead of numpy arrays to save space
        for j in range(temp_binary_masks.shape[0]):
            outfile_binary_mask = outdir_binary_masks + '/' + str(num_cropped_patches) + '__' + str(j) + '_binary_mask.png'     
            imageio.imwrite(outfile_binary_mask, temp_binary_masks[j])

        outfile_mask_color = outdir_mask_color + '/' + str(num_cropped_patches) + '_mask_color.png'
        # print(outfile_mask_color)
        # scipy.misc.imsave(outfile_mask_color, temp_mask_color)
        imageio.imwrite(outfile_mask_color, temp_mask_color)

        # print(temp_label_list)
        # plt.figure()
        # plt.imshow(temp_img)
        # plt.figure()
        # plt.imshow(temp_mask)
        # plt.figure()
        # plt.imshow(temp_mask_color)
        # plt.show()

        with open(cropped_patches_filelist, 'a') as f_cropped_patches_filelist:
            f_cropped_patches_filelist.write(str(num_cropped_patches) + '\t' 
                                             + str(temp_annotation_dict['X_cm']) 
                                             + '\t' + str(temp_annotation_dict['Y_cm']) 
                                             + '\t' + str(temp_center_id) 
                                             + '\t' + str(temp_center_checked)
                                             + '\t' + str(temp_label_list[temp_center_id]) + '\n') 
        
        num_cropped_patches += 1

        pbar.update(1)
            
    # print('large_annotation_count: {}'.format(large_annotation_count))
    # print(skipped)
    # print(len(Annotations))

    pbar.close()

    with open(cropped_patches_info, 'a') as f_cropped_patches_info:
        f_cropped_patches_info.write(str(slide_id) + '\t' + str(num_cropped_patches) + '\n')





