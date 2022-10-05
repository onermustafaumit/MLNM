import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, roc_auc_score
import math
import argparse


plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})

def score_fnc(data_arr1, data_arr2):
	auc = roc_auc_score(data_arr1, data_arr2)
	return auc

def BootStrap(data_arr1, data_arr2, n_bootstraps):

	# initialization by bootstraping
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []
	# print(data_arr2)
	# print(data_arr2)

	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.randint(0, len(data_arr2), len(data_arr2))

		if len(np.unique(data_arr1[indices])) < 2:
			# We need at least one sample from each class
			# otherwise reject the sample
			#print("We need at least one sample from each class")
			continue
		else:
			score = score_fnc(data_arr1[indices], data_arr2[indices])
			bootstrapped_scores.append(score)
			#print("score: %f" % score)

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.
		
	# Computing the lower and upper bound of the 95% confidence interval
	# You can change the bounds percentiles to 0.025 and 0.975 to get
	# a 95% confidence interval instead.
	#print(sorted_scores)
	#print(len(sorted_scores))
	#print(int(0.025 * len(sorted_scores)))
	#print(int(0.975 * len(sorted_scores)))
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
	# print(confidence_lower)
	# print(confidence_upper)
	# print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
	return sorted_scores, confidence_lower, confidence_upper




parser = argparse.ArgumentParser(description='Plot ROC and calculate AUROC value.')

parser.add_argument('--processed_predictions_file', default='', help='File containing processed predictions', dest='processed_predictions_file')

FLAGS = parser.parse_args()

# "test_metrics__panda/YYYY_MM_DD__HH_MM_SS__XXX/panda__radboud__karolinska__all__test/test_scores__YYYY_MM_DD__HH_MM_SS__XXX__processed.txt
processed_predictions_file = FLAGS.processed_predictions_file

# read predictions file
data_arr = np.loadtxt(processed_predictions_file, delimiter='\t',comments='#',dtype=str)
slide_id_arr = data_arr[:,0]
data_provider = data_arr[:,1]
isup_grade = np.asarray(data_arr[:,2],dtype=int)
min_score_malignant = np.asarray(data_arr[:,3],dtype=float)
max_score_malignant = np.asarray(data_arr[:,4],dtype=float)
avg_score_malignant = np.asarray(data_arr[:,5],dtype=float)


fpr, tpr, _ = roc_curve(isup_grade>0, max_score_malignant, pos_label=1)
roc_auc = auc(fpr, tpr)
# print(roc_auc)

sorted_scores, scores_lower, scores_upper = BootStrap(isup_grade>0, max_score_malignant, n_bootstraps=2000)


# title_text = 'AUROC = {:.3f} (CI: {:.3f} - {:.3f})'.format(roc_auc, scores_lower, scores_upper)
title_text = 'AUROC = {:.3f} ({:.3f} - {:.3f})'.format(roc_auc, scores_lower, scores_upper)
print(title_text)

fig, ax = plt.subplots(figsize=(3,3))
ax.plot(fpr, tpr, lw=2, alpha=1., color='k')
# ax.plot(fpr, tpr, color='k', lw=2)
# ax.plot([0, 1], [0, 1], 'k--', lw=1)



ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim((-0.05,1.05))
ax.set_xticks(np.arange(0,1.05,0.2))
# ax.set_xticklabels('')
ax.set_ylim((-0.05,1.05))
ax.set_yticks(np.arange(0,1.05,0.2))
# ax.set_yticklabels('')
ax.set_axisbelow(True)
ax.grid(color='gray') #, linestyle='dashed')
ax.set_title(title_text)
# ax.legend(framealpha=1.)




fig.tight_layout()
# fig.subplots_adjust(left=0.15, bottom=0.12, right=0.98, top=0.98, wspace=0.20 ,hspace=0.20 )
fig_filename = '{}__roc.pdf'.format(processed_predictions_file[:-4])
fig.savefig(fig_filename, dpi=300)

fig_filename = '{}__roc.png'.format(processed_predictions_file[:-4])
fig.savefig(fig_filename, dpi=300)

plt.show()

