import argparse
import matplotlib.pyplot as plt
import numpy as np

def moving_avg_filter(data_arr, w):
    data_arr_cumsum = np.cumsum(data_arr)
    data_arr_cumsum[w:] = (data_arr_cumsum[w:] - data_arr_cumsum[:-w])
    data_arr_filtered = data_arr_cumsum[w-1:]/w
    
    return data_arr_filtered

plt.rcParams.update({'font.size':8, 'font.family':'Arial'})

parser = argparse.ArgumentParser('Plot the loss vs iteraation and accuracy vs iteration for givern data file')

parser.add_argument('--data_file', default='', help='Data file filepath', dest='data_file') # loss_data/step_acc_loss_metrics__2021_06_08__11_23_14 (copy)
parser.add_argument('--step_size', default=1, type=int, help='Step size', dest='step_size')
parser.add_argument('--filter_size', default=1, type=int, help='Filter size', dest='filter_size')

FLAGS = parser.parse_args()

w = FLAGS.filter_size

data_arr = np.loadtxt(FLAGS.data_file, dtype='float', comments='#', delimiter='\t')

fig, ax = plt.subplots(figsize=(4,3))

temp_ind = 2 # total loss

# steps = np.arange(len(data_arr[:, 0]))+1
steps = data_arr[:,0]
temp_train = data_arr[:,temp_ind]

if w > 1:
    steps = steps[w-1:]
    temp_train = moving_avg_filter(temp_train, w)

ind_start = 0
ind_step = FLAGS.step_size
ind_end = len(steps) + 1
# ind_end = 600

ax.plot(steps[ind_start:ind_end:ind_step], temp_train[ind_start:ind_end:ind_step], color='r', linestyle='-',label='Total Loss')

# plt.title('Training Loss for Mask RCNN vs epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.grid(linestyle='--')
plt.legend()

fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.98, wspace=0.20 ,hspace=0.20 )
fig.savefig('{}.png'.format(FLAGS.data_file[:-4]),dpi=200,transparent=False)

plt.show()
