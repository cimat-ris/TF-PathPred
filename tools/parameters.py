import tensorflow as tf
import numpy as np

from tools.optimizer import CustomSchedule

pars = np.load("./tools/parameters.npy")
Tobs,Tpred,d_model,num_heads,num_layers,num_modes,dff = np.array(pars[:-1], dtype = int)
dropout_rate = pars[-1]

# Tobs = 8 
# Tpred = 12

# d_model = 128
# num_heads = 8
# num_layers = 6
# num_modes = 20
# dff = 512
# dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(d_model)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True, reduction='none')