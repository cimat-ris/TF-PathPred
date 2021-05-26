from flask import Flask, request, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
import numpy as np
import json

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return redirect(url_for('base'))

@app.route("/base/", methods=["GET", "POST"])
def base():
	return render_template('base.html')

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
	
	if mode == "view":
		return render_template('parameters.html',
								Tobs = pars[0], Tpred = pars[1],
								d_model = pars[2], num_head = pars[3],
								num_layers = pars[4], num_modes = pars[5],
								dff = pars[6], dropout_rate = pars[7])
	if mode == "change":
		return render_template('change_parameters.html',
								Tobs = pars[0], Tpred = pars[1],
								d_model = pars[2], num_head = pars[3],
								num_layers = pars[4], num_modes = pars[5],
								dff = pars[6], dropout_rate = pars[7])


@app.route("/act_pars/", methods=["GET", "POST"])
def act_pars():
	if request.method == 'POST':
		user = request.form
		ls = []
		for a in user: ls.append(user[a])
		ls = np.array(ls, dtype = "float32")
		np.save("parameters.npy", ls)
	return redirect(url_for('parameters', mode = "view"))

@app.route("/train_h/", methods=["GET", "POST"])
def train_h():
	if request.method == 'POST':
		user = request.form
		ls = list(user)
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

	return render_template('train_progress.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)