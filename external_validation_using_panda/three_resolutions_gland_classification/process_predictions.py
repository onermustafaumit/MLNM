import numpy as np
import sys
import argparse


parser = argparse.ArgumentParser(description='Test a trained multi-resolution gland classification model on PANDA dataset.')

parser.add_argument('--predictions_file', default='', help='File containing predictions obtained from inference', dest='predictions_file')

FLAGS = parser.parse_args()

# "test_metrics__panda/YYYY_MM_DD__HH_MM_SS__XXX/panda__radboud__karolinska__all__test/test_scores__YYYY_MM_DD__HH_MM_SS__XXX.txt
predictions_file = FLAGS.predictions_file



dataset_file = '../panda_dataset/{}.txt'.format(predictions_file.split('/')[-2])

# read dataset file
data_arr = np.loadtxt(dataset_file, delimiter='\t',comments='#',dtype=str)
slide_id_arr = data_arr[:,0]
provider_arr = data_arr[:,1]
isup_grade_arr = np.asarray(data_arr[:,-2],dtype=int)

info_dict = dict()
for i,slide_id in enumerate(slide_id_arr):
	info_dict[slide_id] = { 'provider':provider_arr[i],
							'isup_grade':isup_grade_arr[i]
							}


# read predictions file
data_arr = np.loadtxt(predictions_file, delimiter='\t',comments='#',dtype=str)
slide_id_arr = data_arr[:,0]
score_arr = np.asarray(data_arr[:,-1],dtype=float)

unique_slide_ids = np.unique(slide_id_arr)
num_slides = len(unique_slide_ids)

out_file = '{}__processed.txt'.format(predictions_file[:-4])
with open(out_file,'w') as f_out_file:
	f_out_file.write('# image_id\tdata_provider\tisup_grade\tmin_score_malignant\tmax_score_malignant\tavg_score_malignant\n')\


for i,slide_id in enumerate(unique_slide_ids):
	print('slide {}/{}: {}'.format(i+1,num_slides,slide_id))

	temp_data_provider = info_dict[slide_id]['provider']
	temp_isup_grade = info_dict[slide_id]['isup_grade']


	temp_indices = np.where(slide_id_arr==slide_id)[0]
	# print(temp_indices)
	
	temp_score_arr = score_arr[temp_indices]

	temp_min_score_malignant = np.amin(temp_score_arr)
	temp_max_score_malignant = np.amax(temp_score_arr)
	temp_mean_score_malignant = np.mean(temp_score_arr)


	with open(out_file,'a') as f_out_file:
		f_out_file.write('{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(slide_id,temp_data_provider,temp_isup_grade,temp_min_score_malignant,temp_max_score_malignant,temp_mean_score_malignant))






