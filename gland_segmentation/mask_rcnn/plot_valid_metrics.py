import argparse
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

def moving_avg_filter(data_arr, w):
    data_arr_cumsum = np.cumsum(data_arr)
    data_arr_cumsum[w:] = (data_arr_cumsum[w:] - data_arr_cumsum[:-w])
    data_arr_filtered = data_arr_cumsum[w-1:]/w
    
    return data_arr_filtered


parser = argparse.ArgumentParser('Plot the loss vs iteraation and accuracy vs iteration for givern data file')

parser.add_argument('--data_file', default='', help='Data file filepath', dest='data_file') # 'loss_data/step_valid_metrics__2021_06_08__11_23_14 (copy).txt'
parser.add_argument('--step_size', default=1, type=int, help='Step size', dest='step_size')
parser.add_argument('--filter_size', default=1, type=int, help='Filter size', dest='filter_size')

FLAGS = parser.parse_args()

w = FLAGS.filter_size

data_arr = np.loadtxt(FLAGS.data_file, dtype='float', comments='#', delimiter='\t')

metric_list = ['average precision @ IoU=0.50 | area=all | maxDets=100', 'average recall @ IoU=0.50 | area=all | maxDets=100']
ind_dict = {'Bounding boxes': [2,13], 'Segmentation masks': [15,26]} 
color_list = ['r','g']

fig, ax = plt.subplots(1,2, figsize=(8,3), sharey=True)

for i, (key, value) in enumerate(ind_dict.items()):
    temp_ind_list = value 

    for j, metric in enumerate(metric_list):

        temp_ind = temp_ind_list[j]
        # steps = np.arange(len(data_arr[:, 0]))+1
        steps = data_arr[:, 0]
        temp_valid = data_arr[:, temp_ind]

        if w > 1:
            steps = steps[w-1:]
            temp_valid = moving_avg_filter(temp_valid, w)

        ind_start = 0
        ind_step = FLAGS.step_size
        ind_end = len(steps) + 1
        # ind_end = 600

        ax[i].plot(steps[ind_start:ind_end:ind_step], temp_valid[ind_start:ind_end:ind_step], color=color_list[j], linestyle='-', label='\n'.join(wrap(metric, 20)))
        # set formatting to be the same for both subplots
        # ax[i].tick_params(axis='both', which='both', labelsize=10)
        
        # set x-axis ticks to be visible for second subplot
        ax[i].yaxis.set_tick_params(labelleft=True)
    
    ax[i].set_title(key)
    ax[i].set_xlabel('epoch')
    ax[i].grid(linestyle='--')
    ax[i].legend(loc='lower right', fontsize='small')

#ax[0].set_ylabel('value')
ax[0].set_ylabel('Precision or Recall')
ax[1].set_ylabel('Precision or Recall')
# fig.suptitle('Validation Metrics for Mask RCNN')
# plt.tight_layout()

fig.subplots_adjust(left=0.07, bottom=0.15, right=0.99, top=0.91, wspace=0.20 ,hspace=0.20 )
fig.savefig('{}.png'.format(FLAGS.data_file[:-4]),dpi=200,transparent=True)

plt.show()

