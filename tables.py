from train_TF import train_model
from test_TF import test_model
import numpy as np
import argparse

if __name__ == '__main__':

	# Parser arguments
	parser = argparse.ArgumentParser(description='Train transformer')
	parser.add_argument('--root-path', '--root',
	                    default='./',
	                    help='path to folder that contain dataset')
	args = parser.parse_args()

	datasets = ['ETH-univ','ETH-hotel', 'UCY-zara1', 'UCY-zara2', 'UCY-univ3']
	ADE = {}
	FDE = {}
	for i in range(len(datasets)):
		training_names = datasets.copy()
		test_name = [datasets[i]]
		del training_names[i]

		# print(f"model for {test_name[0]}")
		# train_model(training_names,test_name,args.root_path,50)
		ade,fde = test_model(test_name,args.root_path)
		ADE[test_name[0]] = ade
		FDE[test_name[0]] = fde

	for name in datasets:
		print(f"for {name}, ADE: {np.mean(ADE[name])} and FDE: {FDE[name]}")