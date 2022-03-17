from train_TF import train_model
from test_TF import test_model
import numpy as np
import argparse
import math
import os

if __name__ == '__main__':

	# Parser arguments
	parser = argparse.ArgumentParser(description= 'Experimment for the ')
	parser.add_argument('--root-path', '--root',
	                    default='./',
	                    help='path to folder that contain dataset')
	parser.add_argument('--dataset', '--ds',
	                    default='',
	                    help='path to folder that contain dataset')
	parser.add_argument('--beta', '--bt',
	                    default='0',
	                    help='path to folder that contain dataset')
	args = parser.parse_args()

	if not os.path.isfile('./generated_data/beta_experiment/results.npy'):
		table = np.zeros((5,10))
		table[:] = np.NaN
		np.save('./generated_data/beta_experiment/results.npy', table)


	#------------- Training and updating results in the table of experiment -------------

	table = np.load('./generated_data/beta_experiment/results.npy')

	datasets = ['ETH-univ','ETH-hotel', 'UCY-zara1', 'UCY-zara2', 'UCY-univ3']
	betas = np.array([0,0.3,0.5,1,-1])

	try:

		test_index = int(args.dataset)
		beta_index = int(args.beta)

		training_names = datasets.copy()
		test_name = [datasets[test_index]]
		del training_names[i]
		beta = betas[beta_index]

		print(f"starting training for {test_name}")
		train_model(training_names,test_name,args.root_path,beta = beta, epochs = 50)
		ade,fde = test_model(test_name,args.root_path)
		ade_average = np.mean(ade)
		fde_average = np.mean(fde)
		table[test_index,beta_index*2] = ade_average
		table[test_index,beta_index*2+1] = fde_average
		np.save('./generated_data/beta_experiment/results.npy', table)

	except:

		print("training did not start either because not demanded or because of poor parameter handling")

	#--------------------- Printing results in the table of experiment --------------------

	# This generates latex code for the table to be copied directly
	print("Printing current values")
	for i in range(5):
		s = datasets[i]
		for j in range(5):
			s += " & " 
			if math.isnan(table[i,2*j]):
				s += "NC"
			else:
				s += str(table[i,2*j])
			s += "/"
			if math.isnan(table[i,2*j+1]):
				s += "NC"
			else:
				s += str(table[i,2*j+1])
		s += " \\\\"
		print(s)