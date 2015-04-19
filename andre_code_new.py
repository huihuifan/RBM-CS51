from __future__ import absolute_import, print_function, unicode_literals, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import random as rand

class RBM(object):

	def __init__(self, data, num_hidden, learning_rate):
		# from input
		self.data = data
		self.num_hidden = num_hidden
		self.learning_rate = learning_rate
		# created
		self.hidden_states = np.random.rand(self.num_hidden + 1)
		self.hidden_states[0] = 1
		self.num_vis = data.shape[1]
		self.num_data = data.shape[0]
		self.visible_states = np.concatenate((np.ones((self.num_data,1)), self.data),axis=1)
		self.weights = np.random.rand(self.visible_states.shape[1], self.num_hidden + 1)

	# Debug Function
	def _debug(self):
		print(agent._sample_h_given_v(self.visible_states[2]))
		return

	# Logistic Sigmoid Function
	def _sigmoid(self,x):
		return 1/(1+np.exp(-x))

	# # Binomial Sampling
	# def _binom_sample(probs):
	# 	sample = [np.random.binomial(n=1, p=i) for i in prob]
	# 	return np.array(sample)

    # Predict hidden given visible
	def _sample_h_given_v(self, v):
		prob = self._sigmoid(np.dot(self.weights.T, v.T))
		sample = [np.random.binomial(n=1, p=i) for i in prob]
		return np.array(sample)

    # Predict visible given hidden
	def _sample_v_given_h(self, h):
		prob =  self._sigmoid(np.dot(self.weights, h))
		sample = [np.random.binomial(n=1, p=i) for i in prob]
		return np.array(sample) 

toy_data = np.zeros((20,60))
agent = RBM(toy_data,2,0.1)
#print(agent._sample_h_given_v())
agent._debug()



