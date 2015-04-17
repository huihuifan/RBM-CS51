'''
CS51 - Final Project - Spring 2015

Team members: Andre T. Nguyen, Angela Fan, Jianchuan Zeng, Vincent Nguyen
Contact: andrenguyen@college.harvard.edu

Project: Restricted Boltzmann Machine
'''

# Import modules.
from __future__ import absolute_import, print_function, unicode_literals, division
import matplotlib.pyplot as plt
import numpy as np

'''
Restricted Boltzmann Machine Agent Class

DOCUMENTATION HERE
'''

class RBM(object):

	# Init Function
	def __init__(self, visible, hidden, activation_func, learning_rate):
		# inputs
		self.num_visible = len(visible)
		self.num_hidden = len(hidden)
		self.visible = visible 
		self.hidden = hidden
		self.visible = np.insert(visible, 0, 1)  # add bias
		self.hidden = np.insert(hidden, 0, 1)  # add bias
		self.learning_rate = learning_rate
		self.activation_func = activation_func
		# initialize internal variables 
		self.weights = np.random.rand((self.num_visible+1,self.num_hidden+1)) + 0.01  # includes bias index 0

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

    # Derivative of the log probability of a training vector with respect to a weight
	def deriv_log_prob(self):
		return TODO

	# Stochastic steepest ascent in the log probability of the training data
	def delta(self):




