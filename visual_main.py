from flask import Flask, request, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
import numpy as np
import json

import threading, os
from subprocess import call 

def render(main, tagg, refresh = "no", **kwargs):

	return render_template('base.html', main = main, tagg = tagg, refresh = refresh, **kwargs)


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return redirect(url_for('training'))

@app.route("/training/", methods=["GET", "POST"])
def training():
	return render_template('base.html', main = "training.html", tagg = "Training")

def get_pars():
	pars = np.load("./tools/parameters.npy")
	aux = pars[-1]
	pars = pars[:-1]
	pars = np.array(pars, dtype = int)
	pars = np.array(pars, dtype = str)
	aux = np.array([aux], dtype = str)
	pars = np.concatenate([pars,aux])

	return pars


@app.route("/parameters/<mode>", methods=["GET", "POST"])
def parameters(mode = "view"):
	pars = get_pars()
	aux = []
	for p in pars: aux.append(str(p))
	print(aux)

	headers = ("Parameter", "Value")
	params = ["Size of observations","Size of predictions",
			"Size of embedding","Number of attention heads","Number of sublayers",
			"Number of modes","Number of Neurons","Dropout rate"]
	
	
	if mode == "view":
		data = []
		for i in range(8):
			data.append((params[i],aux[i]))
		data = tuple(data)
		return render("parameters.html","Parameters", headers = headers, data = data)
	if mode == "change":
		ids = ["Tobs", "Tpred", "d_model", "num_heads",
			"num_layers", "num_modes", "dff", "dropout_rate"]

		data = []
		for i in range(8):
			data.append((params[i],aux[i],ids[i]))
		data = tuple(data)

		return render("change_parameters.html", "Parameters",
					headers = headers, data = data, ids = ids)


@app.route("/act_pars/", methods=["GET", "POST"])
def act_pars():
	if request.method == 'POST':
		user = request.form
		ls = []
		for a in user: ls.append(user[a])
		ls = np.array(ls, dtype = "float32")
		np.save("./tools/parameters.npy", ls)
	return redirect(url_for('parameters', mode = "view"))

@app.route("/train_h/", methods=["GET", "POST"])
def train_h():
	if request.method == 'POST':
		user = request.form
		ls = list(user)
		np.save("./static/temp/progress.npy",np.array([1,0]))
		def thread_second():
			call(["python", "train_TF.py"])
		processThread = threading.Thread(target=thread_second)  
		processThread.start()
		
		if len(ls)>2:
			epochs = int(user[ls[0]])
			test_name = user[ls[0]]
			training_names = list(user)[2:]
			print(test_name)
			print(training_names)
		else:
			print("no dataset for training selected")

	return redirect(url_for('train_progress'))

@app.route("/train_progress/", methods=["GET", "POST"])
def train_progress():
	status, progress = np.load("./static/temp/progress.npy")
	if not status == 1:
		return redirect(url_for('training'))

	# return render_template('base.html', 	main = 'train_progress.html', tagg = "Training", refresh = "yes",
	# 						progress = progress)
	return render('train_progress.html', 'Training', refresh = 'yes', progress = progress)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)