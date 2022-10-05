import os
import numpy as np
import glob
import torch
import torch.utils.data
import imageio
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
	"""
    Creates a dataset for inference using the trained instance segmentation model. 
    Only H&E patches are required for inference. 

    Args:
        root (str): path to directory storing cropped H&E patches
        slide_list_filename (str): path to text file containing slide ids that 
            will be used to create the dataset

    """

	def __init__(self, root=None, slide_list_filename=None):
		self.root = root
		self._num_slides, self._img_paths = self.read_slide_list(slide_list_filename)
		self._num_imgs = len(self._img_paths)

	@property
	def num_slides(self):  # num of slides
		return self._num_slides

	@property
	def num_imgs(self):  # num of images
		return self._num_imgs

	@property
	def img_paths(self):
		return self._img_paths

	def __len__(self):
		return len(self._img_paths)

	def read_slide_list(self, slide_list_filename):
		img_path_list = list()

		slide_id_arr = np.loadtxt(slide_list_filename, comments='#', delimiter='\t', dtype=str)
		slide_id_arr = slide_id_arr.reshape((-1,))
		num_slides = slide_id_arr.shape[0]

		for i in range(num_slides):
			slide_id = slide_id_arr[i]
			img_folder_path = os.path.join(self.root, slide_id, 'img')
			temp_img_path_list = glob.glob( '{}/*[0-9].png'.format(img_folder_path))
			temp_img_path_list = sorted(temp_img_path_list)

			img_path_list += temp_img_path_list

		return num_slides, img_path_list

	def __getitem__(self, idx):
		img_path = self._img_paths[idx]
		# print('image path: {}'.format(img_path))

		img = Image.open(img_path).convert("RGB")
		img = F.center_crop(img, (362,362)) # output_size: (height, width)
		img = F.to_tensor(img)

		return img, img_path



def collate_fn(batch):
	return tuple(zip(*batch))



