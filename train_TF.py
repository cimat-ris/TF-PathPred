import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)


from tools.opentraj_benchmark.all_datasets import get_trajlets


from tools.trajectories import obs_pred_trajectories, obs_pred_rotated_velocities, convert_to_traj_with_rotations, convert_to_traj
from tools.parameters import *


from tools.transformer.transformer import Transformer
from tools.transformer.masking import create_look_ahead_mask
from tools.transformer.training import loss_function, accuracy_function, train_step

def train_model(training_names, test_name, path, EPOCHS = 50):
	# trajlets is a dictionary of trajectories, keys are the datasets names
	trajlets = get_trajlets(path, training_names)
    # Xm and Xp will hold the observations and paths-to-predict, respectively
    # Dimensions: NxTobsx2
	Xm     = np.zeros([1,Tobs-1,2], dtype = "float32")
	Xp     = np.zeros([1,Tpred,2],  dtype = "float32")
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

	checkpoint_path = f"./generated_data/checkpoints/train/{test_name[0]}"

	ckpt = tf.train.Checkpoint(transformer=transformer,
									optimizer=optimizer)

	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

	# if a checkpoint exists, restore the latest checkpoint.
	if ckpt_manager.latest_checkpoint:
	  #ckpt.restore(ckpt_manager.latest_checkpoint)
	  print ('Latest checkpoint restored!!')


	train_dataset = {"observations":[],"predictions":[]}
	# Form the training dataset
	for i in range(len(Xp)):
		train_dataset["observations"].append(Xm[i])
		train_dataset["predictions"].append(Xp[i])
	# Get the necessary data into a tf Dataset
	train_data = tf.data.Dataset.from_tensor_slices(train_dataset)
	# Form batches
	batched_train_data   = train_data.batch(32)
	num_batches_per_epoch= batched_train_data.cardinality().numpy()

	train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

	train_loss_results   = []

	# Main training loop
	for epoch in range(EPOCHS):
		start = time.time()
		train_accuracy.reset_states()
		total_loss = 0
		# Iterate over batches
		for (id_batch, batch) in enumerate(batched_train_data):
			if epoch < 2:
				batch_loss = train_step(batch["observations"], batch["predictions"], transformer, optimizer, train_accuracy, burnout = True)
			else:
				batch_loss = train_step(batch["observations"], batch["predictions"], transformer, optimizer, train_accuracy)
			total_loss+=batch_loss
			print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, id_batch, batch_loss, train_accuracy.result()))
		total_loss = total_loss/num_batches_per_epoch
		train_loss_results.append(total_loss)
		if (epoch + 1) % 6 == 0:
			ckpt_save_path = ckpt_manager.save()
			print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
																ckpt_save_path))

		print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
														total_loss,
														train_accuracy.result()))

		print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
	fig,ax = plt.subplots(1)
	plt.margins(0, 0)
	plt.plot(train_loss_results)
	plt.show()
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


	transformer = train_model(training_names,test_name,args.root_path,35)
