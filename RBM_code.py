'''
CS51 - Final Project - Spring 2015

Team members: Andre T. Nguyen, Angela Fan, George Zeng, Vincent Nguyen
Contact: andrenguyen@college.harvard.edu

Project: Restricted Boltzmann Machine
'''

# Import modules
from __future__ import absolute_import, print_function, unicode_literals, division
import matplotlib.pyplot as plt
import numpy as np

'''
Restricted Boltzmann Machine Agent Class

DOCUMENTATION HERE:

PAPER:
http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

NOTES:
- currently expects binary hidden and visible units (might want to change as extension)
- we still need to add regularization (as extension)
- basically, all the stuff after part 2 of paper are possible extensions
'''

class RBM(object):

	'''
	Init Function
	'''
	def __init__(self, num_visible, num_hidden, activation_func, learning_rate):
		# Inputs
		self.num_visible = num_visible
		self.num_hidden = num_hidden
		self.visible = np.random.rand(num_visible)
		self.hidden = np.random.rand(num_hidden)
		self.visible = np.insert(self.visible, 0, 1)  # add bias
		self.hidden = np.insert(self.hidden, 0, 1)  # add bias
		self.learning_rate = learning_rate
		self.activation_func = activation_func
		# Initialize internal variables 
		self.weights = np.random.rand((self.num_visible+1,self.num_hidden+1)) + 0.01  # includes bias index 0


	'''
	Private Methods
	'''

	# Hopfield Energy Function
	def energy(self, visible_states, hidden_states):  
		return (- np.dot(visible_states, self.weights[:,0]) - np.dot(hidden_states, self.weights[0,:]) - np.dot(hidden_states[1:], np.dot(W[1:,1:],visible_states[1:])))

	# Logistic Sigmoid Function
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	# Partition Function
	def Z(self):
		return TODO

	# Probability of Visible/Hidden Pair 
	def probability(self, visible_states, hidden_states):  
		return TODO

	# Probability of Visible
	def probability_visible(self, visible_states):  
		return TODO

	# Get unbiased sample of <data>
	... TODO

	# Get unbiased sample of <model>
	... TODO

    # Derivative of the log probability of a training vector with respect to a weight
	def deriv_log_prob(self, visible_states, i, j):
		return TODO

	# Stochastic steepest ascent in the log probability of the training data
	def delta(self, i, j, visible_states):
		return TODO

	# Gibbs Sampler
	... TODO


	'''
	Public Methods
	'''

	# Train learner
	def train(self, data):
		return TODO

	# Predict Hidden to Visible
	def predict_H2V(self, hidden_states):
		return TODO

	# Predict Visible to Hidden
	def predict_V2H(self, visible_states):
		return TODO




'''
Run RBM on Real Data
'''

... TODO



