from IPython import embed
import glob
import pandas as pd
import pickle
import os
import torch
from torch import nn
from torch.utils import data
import random
import numpy as np


'''for sanity check'''
def naive_social(p1_key, p2_key, all_data_dict):
	if abs(p1_key-p2_key)<4:
		return True
	else:
		return False


def find_min_time(t1, t2):
	'''given two time frame arrays, find then min dist (time)'''
	min_d = 9e4
	t1, t2 = t1[:8], t2[:8]

	for t in t2:
		if abs(t1[0]-t)<min_d:
			min_d = abs(t1[0]-t)

	for t in t1:
		if abs(t2[0]-t)<min_d:
			min_d = abs(t2[0]-t)

	return min_d


def find_min_dist(p1x, p1y, p2x, p2y):
	'''given two time frame arrays, find then min dist'''
	min_d = 9e4
	p1x, p1y = p1x[:8], p1y[:8]
	p2x, p2y = p2x[:8], p2y[:8]

	for i in range(len(p1x)):
		for j in range(len(p1x)):
			if ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5 < min_d:
				min_d = ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5

	return min_d


def social_and_temporal_filter(p1_key, p2_key, all_data_dict, time_thresh=48, dist_tresh=100):
	p1_traj, p2_traj = np.array(all_data_dict[p1_key]), np.array(all_data_dict[p2_key])  # each traj is of shape (T, 4), with 4 denoting [person_id, frame_id, x, y]
	p1_time, p2_time = p1_traj[:,1], p2_traj[:,1]
	p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
	p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]

	if find_min_time(p1_time, p2_time)>time_thresh:
		return False
	if find_min_dist(p1_x, p1_y, p2_x, p2_y)>dist_tresh:
		return False

	return True


def mark_similar(mask, sim_list):
	for i in range(len(sim_list)):
		for j in range(len(sim_list)):
			mask[sim_list[i]][sim_list[j]] = 1  # Todo: the mask is not for paired agent? weird?


# save all train / validate files in a .pickle file
def collect_train_data(set_name, batch_size=512, time_thresh=48, dist_tresh=100, scene=None, verbose=True, root_path="./"):
	assert set_name in ['train', 'validate']

	if set_name == 'train':
		rel_path = 'train'
	else:
		rel_path = 'validate'

	full_dataset = []
	full_masks = []

	current_batch = []
	mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

	current_size = 0  # current_size acts in a batch as current pedestrian ID
	social_id = 0
	part_file = '/{}.txt'.format('*' if scene == None else scene)  # .txt or .csv

	for file in glob.glob(root_path + rel_path + part_file):
		data = np.loadtxt(fname=file, delimiter=',')

		data_by_personId = {}  # personId : a list of [person_id, frame_id, x, y]
		for frame_id, person_id, x, y in data:
			frame_id = (int)(frame_id)
			person_id = (int)(person_id)
			if person_id not in data_by_personId.keys():
				data_by_personId[person_id] = []
			data_by_personId[person_id].append([person_id, frame_id, x, y])

		# delete trajectories with less than 20 steps
		for person_id in list(data_by_personId.keys()):
			traj = data_by_personId[person_id]
			if len(traj) < 20:
				print('Traj of person_id {} has an incomplete traj with less than 20 steps.'.format(person_id))
				del data_by_personId[person_id]

		# Let's say we have frame number 0 to 21.
		# Then we have the following three valid sequences.
		# The first sequence is  <observing 0-7 and predict 8-19>
		# The second sequence is <observing 1-8 and predict 9-20>
		# The last sequence is   <observing 2-9 and predict 10-21>
		data_by_id = {}  # segID: a list of [person_id, frame_id, x, y] with exactly 20 steps
		segId = 0
		for person_id in list(data_by_personId.keys()):
			traj = data_by_personId[person_id]  # a list with >= 20 elements
			num_steps = len(traj)
			assert num_steps >= 20

			for initial_step_index in range(0, num_steps - 20 + 1):
				data_by_id[segId] = traj[initial_step_index : initial_step_index + 20]
				segId += 1

		# store data_by_id batch by batch
		all_data_dict = data_by_id.copy()
		if verbose:
			print("Total People: ", len(list(data_by_id.keys())))
		while len(list(data_by_id.keys()))>0:  # when there exists one or more pedestrians
			related_list = []
			curr_keys = list(data_by_id.keys())  # curr_keys contains all the ids of remaining pedestrians

			if current_size<batch_size:  # current_size acts in a batch as current pedestrian ID
				pass
			else:
				# store the batch
				full_dataset.append(current_batch.copy())
				mask_batch = np.array(mask_batch)
				full_masks.append(mask_batch[0:len(current_batch), 0:len(current_batch)])  # mask_batch is cropped

				# reset the batch
				current_size = 0
				social_id = 0
				current_batch = []
				mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

			current_batch.append((all_data_dict[curr_keys[0]]))  # add the trajectory of the first pedestrian
			related_list.append(current_size)  # related_list contains the corresponding pedestrian ID
			current_size += 1
			del data_by_id[curr_keys[0]]  # delete the first pedestrian

			# curr_keys[0] represents current (first) pedestrian, curr_keys[i] represents the i-th pedestrian
			for i in range(1, len(curr_keys)):
				if current_size >= batch_size:
					break

				if social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh, dist_tresh):  # if true, current pedestrian and the i-th pedestrian are spatio-temporal neighbors
					current_batch.append((all_data_dict[curr_keys[i]]))  # append neighboring traj
					related_list.append(current_size)  # append the id of the neighboring pedestrian
					current_size+=1  # Todo: current_size is a strange thing?
					del data_by_id[curr_keys[i]]  # this would not influence curr_keys

			mark_similar(mask_batch, related_list)
			social_id +=1

	if len(full_dataset) == 0:
		full_dataset.append(current_batch)
		mask_batch = np.array(mask_batch)
		full_masks.append(mask_batch[0:len(current_batch),0:len(current_batch)])
	return full_dataset, full_masks


# save every test file in a .pickle file
def collect_test_data(set_name, batch_size=512, time_thresh=48, dist_tresh=100, scene=None, verbose=True, root_path="./"):
	assert set_name in ['test']

	part_file = '/{}.txt'.format('*' if scene == None else scene)  # .txt or .csv

	rel_path = 'test'
	lst_test_files = glob.glob(root_path + rel_path + part_file)

	for dir_file in lst_test_files:
		full_dataset = []
		full_masks = []

		current_batch = []
		mask_batch = [[0 for i in range(int(batch_size * 1.5))] for j in range(int(batch_size * 1.5))]

		current_size = 0  # current_size acts in a batch as current pedestrian ID
		social_id = 0

		# dir_file = root_path + rel_path + '/' + file + '.txt'
		file = dir_file.split('/')
		file = file[-1]
		file = file.split('.txt')
		file = file[0]
		data = np.loadtxt(fname=dir_file, delimiter=',')

		data_by_personId = {}  # personId : a list of [person_id, frame_id, x, y]
		for frame_id, person_id, x, y in data:
			frame_id = (int)(frame_id)
			person_id = (int)(person_id)
			if person_id not in data_by_personId.keys():
				data_by_personId[person_id] = []
			data_by_personId[person_id].append([person_id, frame_id, x, y])

		# delete trajectories with less than 20 steps
		for person_id in list(data_by_personId.keys()):
			traj = data_by_personId[person_id]
			if len(traj) < 20:
				print('Traj of person_id {} has an incomplete traj with less than 20 steps.'.format(person_id))
				del data_by_personId[person_id]

		# Let's say we have frame number 0 to 21.
		# Then we have the following three valid sequences.
		# The first sequence is  <observing 0-7 and predict 8-19>
		# The second sequence is <observing 1-8 and predict 9-20>
		# The last sequence is   <observing 2-9 and predict 10-21>
		data_by_id = {}  # segID: a list of [person_id, frame_id, x, y] with exactly 20 steps
		segId = 0
		for person_id in list(data_by_personId.keys()):
			traj = data_by_personId[person_id]  # a list with >= 20 elements
			num_steps = len(traj)
			assert num_steps >= 20

			for initial_step_index in range(0, num_steps - 20 + 1):
				data_by_id[segId] = traj[initial_step_index : initial_step_index + 20]
				segId += 1

		# store data_by_id batch by batch
		all_data_dict = data_by_id.copy()
		if verbose:
			print("Total People: ", len(list(data_by_id.keys())))
		while len(list(data_by_id.keys()))>0:  # when there exists one or more pedestrians
			related_list = []
			curr_keys = list(data_by_id.keys())  # curr_keys contains all the ids of remaining pedestrians

			if current_size<batch_size:  # current_size acts in a batch as current pedestrian ID
				pass
			else:
				# store the batch
				full_dataset.append(current_batch.copy())
				mask_batch = np.array(mask_batch)
				full_masks.append(mask_batch[0:len(current_batch), 0:len(current_batch)])  # mask_batch is cropped

				# reset the batch
				current_size = 0
				social_id = 0
				current_batch = []
				mask_batch = [[0 for i in range(int(batch_size*1.5))] for j in range(int(batch_size*1.5))]

			current_batch.append((all_data_dict[curr_keys[0]]))  # add the trajectory of the first pedestrian
			related_list.append(current_size)  # related_list contains the corresponding pedestrian ID
			current_size += 1
			del data_by_id[curr_keys[0]]  # delete the first pedestrian

			# curr_keys[0] represents current (first) pedestrian, curr_keys[i] represents the i-th pedestrian
			for i in range(1, len(curr_keys)):
				if current_size >= batch_size:
					break

				if social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh, dist_tresh):  # if true, current pedestrian and the i-th pedestrian are spatio-temporal neighbors
					current_batch.append((all_data_dict[curr_keys[i]]))  # append neighboring traj
					related_list.append(current_size)  # append the id of the neighboring pedestrian
					current_size+=1  # Todo: current_size is a strange thing?
					del data_by_id[curr_keys[i]]  # this would not influence curr_keys

			mark_similar(mask_batch, related_list)
			social_id +=1

		if len(full_dataset) == 0:
			full_dataset.append(current_batch)
			mask_batch = np.array(mask_batch)
			full_masks.append(mask_batch[0:len(current_batch),0:len(current_batch)])
		test = [full_dataset, full_masks]
		test_name = "../social_pool_data/test_{}.pickle".format(file)  # + str(b_size) + "_" + str(t_tresh) + "_" + str(d_tresh) + ".pickle"
		with open(test_name, 'wb') as f:
			pickle.dump(test, f)


def generate_pooled_data(b_size, t_tresh, d_tresh, dataset_type, scene=None, verbose=True):
	if dataset_type == 'train':
		full_train, full_masks_train = collect_train_data('train', batch_size=b_size, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)
		train = [full_train, full_masks_train]
		train_name = "../social_pool_data/train_{0}_{1}_{2}_{3}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh)
		with open(train_name, 'wb') as f:
			pickle.dump(train, f)

	if dataset_type == 'test':
		collect_test_data('test', batch_size=b_size, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)

	if dataset_type == 'validate':
		full_validate, full_masks_validate = collect_train_data('validate', batch_size=b_size, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)
		validate = [full_validate, full_masks_validate]
		validate_name = "../social_pool_data/validate_{0}_{1}_{2}_{3}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh)# + str(b_size) + "_" + str(t_tresh) + "_" + str(d_tresh) + ".pickle"
		with open(validate_name, 'wb') as f:
			pickle.dump(validate, f)


def initial_pos(traj_batches):
	batches = []
	for b in traj_batches:  # traj_batches has only one element; b is of shape (2829, 20, 2)
		starting_pos = b[:,7,:].copy()/1000 #starting pos is end of past, start of future. scaled down.  # Todo: why scaled down by 1000?
		batches.append(starting_pos)

	return batches


# store the origins of each batch, for recovering the trajectories
def origin_pos(traj_batches):
	batches = []
	for b in traj_batches:  # traj_batches has only one element; b is of shape (2829, 20, 2)
		origins = b[:, :1, :].copy()  # origins is of shape (2829, 2)
		batches.append(origins)
	return batches


def calculate_loss(x, reconstructed_x, mean, log_var, criterion, future, interpolated_future):
	# reconstruction loss
	RCL_dest = criterion(x, reconstructed_x)

	ADL_traj = criterion(future, interpolated_future) # better with l2 loss

	# kl divergence loss
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return RCL_dest, KLD, ADL_traj


class SocialDataset(data.Dataset):

	def __init__(self, set_name, b_size=4096, t_tresh=60, d_tresh=50, test_file=None, scene=None, id=False, verbose=True):
		'Initialization'
		load_name = None
		if set_name in ['train', 'validate']:
			load_name = "../social_pool_data/{0}_{1}{2}_{3}_{4}.pickle".format(set_name, 'all_' if scene is None else scene[:-2] + scene[-1] + '_', b_size, t_tresh, d_tresh)
		elif set_name == 'test':
			load_name = "../social_pool_data/{0}_{1}.pickle".format(set_name, test_file)
		else:
			print('Incorrect set_name in SocialDataset')

		print(load_name)
		with open(load_name, 'rb') as f:
			data = pickle.load(f)

		traj, masks = data
		traj_new = []

		if id==False:
			for t, m in zip(traj, masks):  # traj is a list that has only one element
				lst_t = []
				for i in t:
					i = np.asarray(i)
					if i.shape != (20, 4):
						print('find incomplete segment!')
						continue
					j = i.reshape((1, 20, 4))
					lst_t.append(j)
				t = np.concatenate(lst_t, axis=0)
				# t = [np.asarray(i).reshape((1, 20, 4)) for i in t]
				# t = np.concatenate(t, axis=0)
				# t = np.asarray(t)  # t is of shape (2829, 20, 4), with axis=2: (person_id, frame_id, x_wcs, y_wcs)
				t = t[:,:,2:]  # t is of shape (2829, 20, 2), with axis=2: (x_wcs, y_wcs)
				t = t.reshape((1, -1, 20, 2))  # # t is of shape (1, 2829, 20, 2),
				traj_new.append(t)

				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=2).copy()  # reverse the order of elements in t along the given axis (temporal axis)
					traj_new.append(reverse_t)
		else:
			for t in traj:
				t = np.array(t)
				traj_new.append(t)

				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)

		# a mask represent the spatio-temporal neighbours for all pedestrians using proximity threshold t_{dist} for distance in space and ensure temporal overlap.
		# the mask encodes crucial information regarding social locality of different trajectories which gets utilized in attention based pooling.
		# the mask is used point-wise to allow pooling only on the spatio-temporal neighbours masking away other pedestrians in the scene.
		# 1: neighbor; 0: non-neighbor
		masks_new = []
		for m in masks: # masks is a list with only one element
			m_shape = m.shape
			m = m.reshape((1, m_shape[0], m_shape[1]))
			masks_new.append(m)  # m is of shape (2829, 2829)

			if set_name=="train":
				#add second time for the reversed tracklets...
				masks_new.append(m)

		# traj_new = np.array(traj_new)  # traj_new is of shape (1, 2829, 20, 2)
		# masks_new = np.array(masks_new)  # of shape (1, 2829, 2829)
		traj_new = np.concatenate(traj_new, axis=0)
		masks_new = np.concatenate(masks_new, axis=0)
		self.trajectory_batches = traj_new.copy()
		self.mask_batches = masks_new.copy()
		self.initial_pos_batches = np.array(initial_pos(self.trajectory_batches)) #for relative positioning  # of shape (1, 2829, 2)
		self.origin_batches = np.array(origin_pos(self.trajectory_batches)).copy()  # for recovering positions  # of shape (1, 2829, 2)
		if verbose:
			print("Initialized social dataloader...")

"""
We've provided pickle files, but to generate new files for different datasets or thresholds, please use a command like so:
Parameter1: batchsize, Parameter2: time_thresh, Param3: dist_thresh
"""

#generate_pooled_data(512,0,100, 'train', verbose=True)
#print('\n\n')
#generate_pooled_data(4096,0,100, 'validate', verbose=True)
#print('\n\n')
#generate_pooled_data(4096,0,100, 'test', verbose=True)