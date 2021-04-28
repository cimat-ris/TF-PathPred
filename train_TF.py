import os
import time
import argparse
import numpy as np
import tensorflow as tf


from opentraj_benchmark.all_datasets import get_trajlets


from tools.trajectories import obs_pred_trajectories, obs_pred_rotated_velocities, convert_to_traj_with_rotations, convert_to_traj
from tools.parameters import *


from transformer.transformer import Transformer
from transformer.masking import create_look_ahead_mask
from transformer.training import loss_function, accuracy_function, train_step

def train_model(training_names, test_name, path, EPOCHS = 50):
	# trajlets is a dictionary of trajectories, keys are the datasets names
	trajlets = get_trajlets(path, training_names)

	Xm = np.zeros([1,Tobs-1,2], dtype = "float32")
	Xp = np.zeros([1,Tpred,2],  dtype = "float32")
	starts = np.array([[0,0]])
	dists  = np.array([])
	mtcs   = np.array([[[0.,0],[0,0]]])

	# Process all the trajectories on the dictionary
	for key in trajlets:

		# Get just the position information
		trajectories = trajlets[key][:,:,:2]
		print("Reading: ",trajectories.shape[0]," trajectories from ",key)
		# Obtain observed and predicted diferences in trajlets
		_, minus, plus, _, _ = obs_pred_rotated_velocities(trajectories,Tobs,Tpred+Tobs)
		# Append the new past parts (minus) and future parts (plus)
		Xm = np.concatenate((Xm,minus), axis = 0)
		Xp = np.concatenate((Xp,plus), axis = 0)
	# Remove first element
	Xm = Xm[1:]
	Xp = Xp[1:]

	Xm = tf.constant(Xm)
	Xp = tf.constant(Xp)

	#------------------------ Training -------------------------
	# Build the model
	transformer = Transformer(d_model, num_layers, num_heads, dff, Tobs, Tpred, num_modes, dropout_rate)

	checkpoint_path = f"./checkpoints/train/{test_name[0]}"

	ckpt = tf.train.Checkpoint(transformer=transformer,
									optimizer=optimizer)

	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

	# if a checkpoint exists, restore the latest checkpoint.
	if ckpt_manager.latest_checkpoint:
	  ckpt.restore(ckpt_manager.latest_checkpoint)
	  print ('Latest checkpoint restored!!')


	train_dataset = []
	# Form the training dataset
	for i in range(len(Xp)):
		train_dataset.append((Xm[i],Xp[i]))

	train_loss     = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

	# Main training loop
	for epoch in range(EPOCHS):
		start = time.time()

		train_loss.reset_states()
		train_accuracy.reset_states()

		# Iterate over batches
		for (batch, (inp, tar)) in enumerate(train_dataset):
			if epoch < 2:
				train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy, burnout = True)
			else:
				train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy)

			if batch % 50 == 0:
				print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
					epoch + 1, batch, train_loss.result(), train_accuracy.result()))

		if (epoch + 1) % 6 == 0:
			ckpt_save_path = ckpt_manager.save()
			print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
																ckpt_save_path))

		print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
														train_loss.result(),
														train_accuracy.result()))

		print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

	return transformer


if __name__=='__main__':

	#------------------------ Parser ---------------------------

	# Parser arguments
	parser = argparse.ArgumentParser(description='Train transformer')
	parser.add_argument('--root-path', '--root',
	                    default='./',
	                    help='path to folder that contain dataset')
	args = parser.parse_args()

	#------------info for training --------------------------------

	training_names = ['ETH-hotel', 'UCY-zara1', 'UCY-zara2', 'UCY-univ3']
	# training_names = ['ETH-univ','ETH-hotel']
	test_name = ['ETH-univ']

	# training_names = ['ETH-univ', 'UCY-zara1', 'UCY-zara2', 'UCY-univ3']
	# test_name = ['ETH-hotel']

	# training_names = ['ETH-univ','ETH-hotel', 'UCY-zara1', 'UCY-univ3']
	# test_name = ['UCY-zara2']

	# training_names = ['ETH-univ','ETH-hotel','UCY-zara2', 'UCY-univ3']
	# test_name = ['UCY-zara1']

	# training_names = ['ETH-univ','ETH-hotel', 'UCY-zara1', 'UCY-zara2']
	# test_name = ['UCY-univ3']


	transformer = train_model(training_names,test_name,args.root_path,12)
