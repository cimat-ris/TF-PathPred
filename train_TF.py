import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tools.opentraj_benchmark.all_datasets import get_trajlets
from tools.trajectories import obs_pred_trajectories, obs_pred_rotated_trajectories, convert_to_traj_with_rotations, \
    convert_to_traj
from tools.parameters import *
from tools.transformer.transformer import Transformer, Transformer_CVAE
from tools.transformer.masking import create_look_ahead_mask
from tools.transformer.training import loss_function, accuracy_function, train_step


#if beta is negative, it means cyclic annealing
def train_model(training_names, test_name, path, beta = 0, EPOCHS=50):
    # trajlets is a dictionary of trajectories, keys are the datasets names
    trajlets = get_trajlets(path, training_names)
    # Xm and Xp will hold the observations and paths-to-predict, respectively
    # Dimensions: N x Tobs x 2
    observations = np.zeros([1, Tobs - 1, 2], dtype="float32")
    groundtruth  = np.zeros([1, Tpred, 2], dtype="float32")

    # Process all the trajectories on the dictionary
    for key in trajlets:
        # Get just the position information
        trajectories = trajlets[key][:, :, :2]
        print("Reading: ", trajectories.shape[0], " trajectories from ", key)
        # Obtain observed and predicted with normalized speeds and rotations in trajlets
        _, minus, plus, _, _ = obs_pred_rotated_trajectories(trajectories, Tobs, Tpred + Tobs)
        # Append the new past parts (minus) and future parts (plus)
        observations = np.concatenate((observations, minus), axis=0)
        groundtruth = np.concatenate((groundtruth, plus), axis=0)
    # Remove first element
    observations = observations[1:]
    groundtruth  = groundtruth[1:]
    observations = tf.constant(observations)
    groundtruth  = tf.constant(groundtruth)

    # ------------------------ Training -------------------------
    # Build the model
    transformer = Transformer(d_model, num_layers, num_heads, dff, num_modes, dropout_rate)

    checkpoint_path = f"./generated_data/checkpoints/train/{test_name[0]}"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_dataset = {"observations": [], "predictions": []}
    # Form the training dataset
    for i in range(len(Xp)):
        train_dataset["observations"].append(observations[i])
        train_dataset["predictions"].append(groundtruth[i])
    # Get the necessary data into a tf Dataset
    train_data = tf.data.Dataset.from_tensor_slices(train_dataset)
    # Form batches
    batched_train_data = train_data.batch(128)
    num_batches_per_epoch = batched_train_data.cardinality().numpy()

    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    train_loss_results = []

    if beta < 0:
        cyclic_begin = int(EPOCHS/2 - M/2)
        cyclic_end = int(EPOCHS/2 + M/2)
    # Main training loop
    for epoch in range(EPOCHS):
        start = time.time()
        train_accuracy.reset_states()
        total_loss = 0
        # Iterate over batches
        for (id_batch, batch) in enumerate(batched_train_data):

            if epoch < 0: #modify the value to determine how many epochs are for burnout
                batch_loss = train_step(batch["observations"], batch["predictions"], transformer, optimizer, beta = beta,
                                        burnout=True)
            else:
                #When beta is negative, its absolute value represents the amount of cycles
                if beta < 0:
                    if epoch < cyclic_begin: beta_aux = 0
                    elif epoch > cyclic_end: beta_aux = 0
                    else: beta_aux = (epoch - cyclic_begin)/M

                #This is when there is no cyclic annealing
                else: beta_aux = beta

                batch_loss = train_step(batch["observations"], batch["predictions"], transformer, optimizer,
                                        beta = beta_aux)
            total_loss += batch_loss
            if id_batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, id_batch, batch_loss,
                                                                             train_accuracy.result()))
        total_loss = total_loss / num_batches_per_epoch
        train_loss_results.append(total_loss)
        if (epoch + 1) % 3 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            total_loss,
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    # PLot
    fig, ax = plt.subplots(1)
    plt.margins(0, 0)
    plt.plot(train_loss_results)
    plt.show()
    return transformer


if __name__ == '__main__':
    # ------------------------ Parser ---------------------------

    # Arguments parser
    parser = argparse.ArgumentParser(description='Train transformer')
    parser.add_argument('--root-path', '--root',
                        default='./',
                        help='path to folder that contain dataset')
    parser.add_argument('--test',
                        default='ETH-univ',
                        help='name of the dataset to test')
    args = parser.parse_args()

    # ------------info for training -------------------------------
    datasets_names = ['ETH-hotel', 'ETH-univ', 'UCY-zara1', 'UCY-zara2', 'UCY-univ3']
    datasets_test = [dataset for dataset in datasets_names if dataset==args.test]
    datasets_train = [dataset for dataset in datasets_names if dataset!=args.test]
    # Train the model
    transformer = train_model(datasets_train, datasets_test, args.root_path, 0, 51)
