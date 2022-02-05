import argparse
import matplotlib.pyplot as plt
import numpy as np

def moving_avg_filter(data_arr, w):
    data_arr_cumsum = np.cumsum(data_arr)
    data_arr_cumsum[w:] = (data_arr_cumsum[w:] - data_arr_cumsum[:-w])
    data_arr_filtered = data_arr_cumsum[w-1:]/w
    
    return data_arr_filtered

plt.rcParams.update({'font.size':8, 'font.family':'Arial'})

parser = argparse.ArgumentParser('Plot the loss vs iteraation and accuracy vs iteration for given data file')

parser.add_argument('--data_file', default='', help='Data file path', dest='data_file')
parser.add_argument('--step_size', default=1, type=int, help='Data file path', dest='step_size')
parser.add_argument('--filter_size', default=1, type=int, help='Data file path', dest='filter_size')
# parser.add_argument('--metric', default='loss', type=str, help='loss or accuracy', dest='metric')

FLAGS = parser.parse_args()

w = FLAGS.filter_size

data_arr = np.loadtxt(FLAGS.data_file, dtype='float', comments='#', delimiter='\t')


legend_list = [r'High (40$\times$)',r'Medium (20$\times$)',r'Low (10$\times$)',r'Low2 (5$\times$)','Resultant']
ind_list = [2,3,4,5,6]

color_list = ['r','g','b','c','k']

steps = data_arr[:, 0]
ind_start = 0
ind_step = FLAGS.step_size
ind_end = len(steps)

fig, ax = plt.subplots(figsize=(4,3))

for i,legend_name in enumerate(legend_list):
    temp_ind = ind_list[i]

    # steps = data_arr[:, 0]
    steps = np.arange(len(data_arr[:, 0]))+1
    temp_train = data_arr[:,temp_ind]
    temp_valid = data_arr[:,temp_ind+6]

    if w > 1:
        steps = steps[w-1:]
        temp_train = moving_avg_filter(temp_train, w)
        temp_valid = moving_avg_filter(temp_valid, w)

    ind_start = 0
    ind_step = FLAGS.step_size
    if ind_step > 1:
        ind_start = ind_step-1
    ind_end = len(steps) + 1

    ax.plot(steps[ind_start:ind_end:ind_step], temp_train[ind_start:ind_end:ind_step], color=color_list[i], linestyle='-',label=str(legend_name+ ' Train'))
    ax.plot(steps[ind_start:ind_end:ind_step], temp_valid[ind_start:ind_end:ind_step], color=color_list[i], linestyle='--', label=str(legend_name+' Validation'))


metric = FLAGS.data_file.split('/')[-1].split('_')[1]
if metric == 'loss':
    # plt.title('Loss vs Epoch')
    plt.ylabel('Loss')
else:
    # plt.title('Accuracy vs Epoch')
    plt.ylabel('Accuracy')

plt.xlabel('Epoch')
plt.grid(linestyle='--')
plt.legend(labelspacing=.1,columnspacing=.5, handletextpad=0.5, fontsize=8)

fig.subplots_adjust(left=0.13, bottom=0.15, right=0.99, top=0.99, wspace=0.20 ,hspace=0.20 )
fig.savefig('{}.png'.format(FLAGS.data_file[:-4]),dpi=200,transparent=False)

plt.show()


