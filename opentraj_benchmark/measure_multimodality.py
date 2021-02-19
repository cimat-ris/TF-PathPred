# -*- coding: utf-8 -*-
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider
from sklearn.mixture import GaussianMixture

import os
import sys
import glob
from crowdscan.loader.loader_eth import loadETH


# Parser arguments
parser = argparse.ArgumentParser(description='Measure multimodality'
                                             'on trajectories.')
parser.add_argument('--dataset', '--data',
                    default='eth', choices=['eth', 'hotel'],
                    help='pick dataset to work on (defalut: "eth")')
parser.add_argument('--root-path', '--root',
                    default='D:/Users/Francisco/Documents/GitLab/'
                            'crowdscan/crowdscan/tests/toy trajectories/',
                    help='path to foldet that contain dataset')
args = parser.parse_args()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.5 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def to_pixel_coordinates(x, y, args):
    # To pixel coordinates
    x = (x - args.min_x) / np.abs(args.max_x - args.min_x)
    y = (y - args.min_y) / np.abs(args.max_y - args.min_y)

    # Multiply by image dimensions
    x = x * args.dim_0
    y = y * args.dim_1

    return x, y


def plot_dataset(dataset, args):
    # Plot reference
    plt.imshow(args.reference)

    # Plot all trajectories
    for idx, traj in enumerate(dataset):
        # Number of trajectories
        if idx > 100:
            break

        # Get x and y values
        x = traj['pos_x']
        y = traj['pos_y']

        # To pixel coordinates
        x, y = to_pixel_coordinates(x, y, args)

        # To list
        x = x.to_list()
        y = y.to_list()

        # Plot trajectory
        plt.plot(y, x, color='green', alpha=1)

        # Plot direction arrows
        for jdx in range(1, len(x)):
            plt.arrow(y[jdx - 1], x[jdx - 1],
                      y[jdx] - y[jdx - 1],
                      x[jdx] - x[jdx - 1],
                      color='green',
                      shape='full',
                      lw=1,
                      length_includes_head=True,
                      head_width=5)

    plt.title('Sample of trajectories')
    plt.show()


def get_sample_at_time(time, dataset, args, plot=True):
    # Sample
    sample_x = []
    sample_y = []

    # Get every trajectory
    for traj in dataset:
        # Get x and y values
        x = traj['pos_x']
        y = traj['pos_y']

        # To pixel coordinates
        x, y = to_pixel_coordinates(x, y, args)

        # To list
        x = x.to_list()
        y = y.to_list()

        # Ajust cubic spline
        n = len(x)

        # Too small for cubic interpolation
        if n <= 3:
            continue

        # Interpolate and parametrize
        t = np.linspace(0, 1, n)
        x_curve = interp1d(t, x, kind='cubic')
        y_curve = interp1d(t, y, kind='cubic')

        # Get sample
        sample_x.append(x_curve(time))
        sample_y.append(y_curve(time))

    # Plot reference
    if plot:
        print('Sample Size', len(sample_x))
        plt.imshow(args.reference)
        plt.scatter(sample_y, sample_x, color='red')
        plt.show()

    return sample_x, sample_y


def find_number_of_modes(X, plot=True):
    # Find number of modes
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full',
                              random_state=0).fit(X)
              for n in n_components]

    if plot:
        plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
        plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.show()

    # Getting number of modes
    index = np.argmin([m.bic(X) for m in models])
    modes = n_components[index]

    return modes


def main():
    # Load dataset
    if args.dataset == 'eth':
        # ETH
        path = os.path.join(args.root_path, 'ETH/seq_eth/reference.png')

        # Read image
        args.reference = plt.imread(path)

        # Get image dimensions
        args.dim_0, args.dim_1, args.dim_2 = args.reference.shape

        # ETH
        root = os.path.join(args.root_path,
                            'ETH/seq_eth/obsmat.txt')

        dataset = loadETH(root, title='ETH')

    elif args.dataset == 'hotel':
        # ETH
        path = os.path.join(args.root_path, 'ETH/seq_hotel/reference.png')

        # Read image
        args.reference = plt.imread(path)

        # Get image dimensions
        args.dim_0, args.dim_1, args.dim_2 = args.reference.shape

        # Hotel
        root = os.path.join(args.root_path,
                            'ETH/seq_hotel/obsmat.txt')

        dataset = loadETH(root, title='Hotel')

    # Get trajectories from dataset
    trajectories = dataset.get_trajectories()

    # Analyze trajectories
    args.max_x, args.max_y = -float('inf'), -float('inf')
    args.min_x, args.min_y = float('inf'), float('inf')

    for traj in trajectories:
        # Get max and min values
        args.max_x = max(args.max_x, max(traj['pos_x']))
        args.max_y = max(args.max_y, max(traj['pos_y']))
        args.min_x = min(args.min_x, min(traj['pos_x']))
        args.min_y = min(args.min_y, min(traj['pos_y']))

    print('args.max_x', args.max_x)
    print('args.max_y', args.max_y)
    print('args.min_x', args.min_x)
    print('args.min_y', args.min_y)

    # Print parameters from args
    print('\nParameters :')
    for key, value in args.__dict__.items():
        if key != 'reference':
            print(str(key), str(value))
    print()

    # Plot ETH
    plot_dataset(trajectories, args)

    # Measure multimodality
    sample_x, sample_y = get_sample_at_time(1.0, trajectories, args)

    # Create data array
    X = np.array([sample_x, sample_y], dtype=np.float32).T

    # Number of modes
    modes = find_number_of_modes(X)
    print('Number of modes with BIC : ', modes)

    # Fit GMM model to sample
    gmm = GaussianMixture(n_components=modes).fit(X)
    labels = gmm.predict(X)
    plt.imshow(args.reference)
    plt.scatter(X[:, 1], X[:, 0], c=labels,
                s=40, cmap='viridis')
    plt.title('Sample at t = 1.0')
    plt.show()

    # Plot GMM
    plot_gmm(gmm, X)
    plt.imshow(args.reference)
    plt.title('Adjust GMM with 3 components')
    plt.show()

    # Detect number of modes for each time
    times = np.arange(0.0, 1.0, .05)
    num_modes = []
    for t in tqdm(times):
        sample_x, sample_y =\
            get_sample_at_time(t, trajectories, args, plot=False)
        X = np.array([sample_x, sample_y], dtype=np.float32).T
        modes = find_number_of_modes(X, plot=False)
        num_modes.append(modes)

        # Plot GMM
        # plot_gmm(gmm, X)
        # plt.imshow(args.reference)
        # plt.show()

    # Plot estimated
    print('Times : ', times)
    print('Number of Modes :', num_modes)

    # Plot number of modes
    plt.plot(times, num_modes)
    plt.xlabel('times')
    plt.ylabel('num_modes')
    plt.title('Number of modes per each t')
    plt.show()


if __name__ == "__main__":
    main()
