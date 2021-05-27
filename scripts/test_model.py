import sys
sys.path.append("../utils")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import argparse
import copy

from models import PECNet
from social_utils import SocialDataset

import matplotlib.pyplot as plt
import numpy as np
from models import *
from social_utils import *
import yaml
import glob


train_num = 4
parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="PECNET_social_model{}.pt".format(train_num))
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample; originally set to 20
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
print(device)


checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)


# the prediction seems to be stochastic
# best_of_n denotes the number of trials in the prediction, for choosing the best prediction
def test(test_file, test_dataset, model, best_of_n = 1):

	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int
	test_loss = 0

	lst_gt = []  # an element is of shape (num_pedestrians_i, 20, 2)
	lst_pred = [] # an element is of shape (num_pedestrians_i, 1, 12, 2)
	with torch.no_grad():
		# i denotes the batch index
		for i, (traj, mask, initial_pos, origins) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches, test_dataset.origin_batches)):
			# origins is of shape (2829, 1, 2)
			print('i = {}'.format(i))
			traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
			x = traj[:, :hyper_params["past_length"], :]  # past_length = 8; x denotes ground true past trajectory; x is of shape (2829, 8, 2)
			original_x_shape = x.shape
			print('original x.shape = {}'.format(original_x_shape))
			y = traj[:, hyper_params["past_length"]:, :]  # y denotes ground true future trajectory; y is of shape (2829, 12, 2)
			y = y.cpu().numpy()
			# reshape the data
			x = x.contiguous().view(-1, x.shape[1]*x.shape[2])
			x = x.to(device)

			future = y[:, :-1, :]
			dest = y[:, -1, :]  # true final destination, even though the trajectory may have a length > 20
			all_l2_errors_dest = []
			all_guesses = []
			for index in range(best_of_n):

				dest_recon = model.forward(x, initial_pos, device=device)  # x is the past trajectories (the first 8 steps) of all agents
				dest_recon = dest_recon.cpu().numpy()
				all_guesses.append(dest_recon)

				l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)  # l2error_sample is of shape (2829, )
				all_l2_errors_dest.append(l2error_sample)

			all_l2_errors_dest = np.array(all_l2_errors_dest)
			all_guesses = np.array(all_guesses)
			# average error
			l2error_avg_dest = np.mean(all_l2_errors_dest)

			# choosing the best guess
			indices = np.argmin(all_l2_errors_dest, axis = 0)

			best_guess_dest = all_guesses[indices,np.arange(x.shape[0]),  :]

			# taking the minimum error out of all guess
			l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))

			# back to torch land
			best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

			# using the best guess for interpolation
			# Todo: interpolated_future contains 12 interpolated steps starting from step 8 and ending with the estimated final destination, even though the actual steps may > 12 steps?
			interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)  # x is the past trajectories (the first eight steps) of all agents; # mask is of shape (2829, 2829)
			interpolated_future = interpolated_future.cpu().numpy()
			best_guess_dest = best_guess_dest.cpu().numpy()

			# final overall prediction
			predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)  # of shape (2819, 24)
			predicted_future = np.reshape(predicted_future, (-1, hyper_params["future_length"], 2))  # future_length = 12
			print('shape of predicted_future = {}'.format(predicted_future.shape))  # (2829, 12, 2)

			# ADE error
			l2error_overall = np.mean(np.linalg.norm(y - predicted_future, axis = 2))

			l2error_overall /= hyper_params["data_scale"]
			l2error_dest /= hyper_params["data_scale"]
			l2error_avg_dest /= hyper_params["data_scale"]

			print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
			print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))

			# reshape the data
			x = x.contiguous().view(-1, original_x_shape[1], original_x_shape[2])
			print('x.shape = {}'.format(x.shape))  # of shape (2829, 8, 2)
			print('y.shape = {}'.format(y.shape))  # of shape (2829, 12, 2)

			#gt_n_20_2 = np.concatenate((x, y), axis=1)
			#gt_n_20_2 += origins  # origins is of shape (2829, 1, 2); recover trajectory by adding back origins
			#lst_gt.append(gt_n_20_2)
			y += origins  # y is of shape (2829, 12, 2), origins is of shape (2829, 1, 2); recover trajectory by adding back origins
			gt_n_1_12_2 = y.reshape((-1, 1, 12, 2))
			lst_gt.append(gt_n_1_12_2)

			predicted_future += origins  # predicted_future is of shape (2829, 12, 2), origins is of shape (2829, 1, 2); recover trajectory by adding back origins
			predicted_future_save = np.reshape(predicted_future, (-1, 1, hyper_params["future_length"], 2))
			print('shape of predicted_future_save = {}'.format(predicted_future_save.shape))  # (2829, 1, 12, 2)
			lst_pred.append(predicted_future_save)

		# save both predicted and gt future trajectory
		#with open('../output_trajs/{}_gt.npy'.format(test_num), 'wb') as f1:
		#	np.save(f1, np.concatenate(lst_gt, axis=0))  # of shape (num_pedestrians, 20, 2)

		#with open('../output_trajs/{}_pred.npy'.format(test_num), 'wb') as f2:
		#	np.save(f2, np.concatenate(lst_pred, axis=0))  # of shape (num_pedestrians, 1, 12, 2)
		gt_n_1_12_2 = np.concatenate(lst_gt, axis=0)
		pred_n_1_12_2 = np.concatenate(lst_pred, axis=0)
		output_n_2_12_2 = np.concatenate((gt_n_1_12_2, pred_n_1_12_2), axis=1)
		with open('../output_trajs/{}.npy'.format(test_file), 'wb') as f:
			np.save(f, output_n_2_12_2)

	return l2error_overall, l2error_dest, l2error_avg_dest


def main():

	# [0] load model
	model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
	model = model.double().to(device)
	model.load_state_dict(checkpoint["model_state_dict"])

	part_file = '/{}.txt'.format('*')  # .txt or .csv
	lst_test_files = glob.glob('../utils/test' + part_file)

	# load dataset
	for dir_test_file in lst_test_files:
		test_file = dir_test_file.split('/')
		test_file = test_file[-1]
		test_file = test_file.split('.txt')
		test_file = test_file[0]
		test_dataset = SocialDataset('test', test_file=test_file, verbose = args.verbose)

		# test_dataset.trajectory_batches is of shape (1, 2829, 20, 2)
		for traj, origins in zip(test_dataset.trajectory_batches, test_dataset.origin_batches):  # traj is of shape (2829, 20, 2)
			o1 = traj[:, :1, :]
			o2 = origins
			print('same origin: {}'.format(np.array_equal(o1, o2)))  # true
			traj -= traj[:, :1, :]  # traj[:, :1, :] is of shape (2829, 1, 2); shift traj such that it starts from origin
			traj *= hyper_params["data_scale"]  # data_scale is changed to 1.0

		# average ade/fde for k=20 (to account for variance in sampling)
		N = args.num_trajectories  # number of generated trajectories
		num_samples = 1
		average_ade, average_fde = 0, 0
		for i in range(num_samples):
			test_loss, final_point_loss_best, final_point_loss_avg = test(test_file, test_dataset, model, best_of_n=N)
			average_ade += test_loss
			average_fde += final_point_loss_best

		print()
		print("Average ADE:", average_ade / num_samples)
		print("Average FDE:", average_fde / num_samples)

main()
