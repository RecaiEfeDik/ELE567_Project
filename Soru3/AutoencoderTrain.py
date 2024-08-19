# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:39:40 2019

@author: Danish
1. Compute pwr combined (Real, Imag), (Extract R & I parts) Generate single distribution, Separate Sending, (R&I)
"""

from keras.datasets import cifar10
from AutoencoderModel import TrainAutoEncoder, Calculate_filters
import tensorflow as tf
import numpy as np

(trainX, _), (testX, _) = cifar10.load_data()

def normalize_pixels(train_data, test_data):
    #convert integer values to float
	train_norm = train_data.astype('float32')
	test_norm = test_data.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

#normalizing the training and test data
x_train, x_test = normalize_pixels(trainX, testX)
#compression_ratios = [0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49]
compression_ratios = [0.06, 0.09, 0.17]

def add_awgn(image, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    
    signal_power = np.mean(image ** 2)
    
    noise_power = signal_power / snr_linear
    
    noise = np.sqrt(noise_power) * np.random.randn(*image.shape)
    
    noisy_image = image + noise
    
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image


noisy_images = []


snr_values = np.random.uniform(low=0, high=25, size=5000)  

for i in range(5000):

    snr_db = snr_values[i]
    noisy_image = add_awgn(x_train[i], snr_db)
    noisy_images.append(noisy_image)


x_train = np.array(noisy_images)


for comp_ratio in compression_ratios:
    tf.keras.backend.clear_session()
    c = Calculate_filters(comp_ratio)
    print('---> System Will Train, Compression Ratio: '+str(comp_ratio)+'. <---')
    _ = TrainAutoEncoder(x_train, x_test, nb_epoch=75, comp_ratio=comp_ratio, batch_size=16, c=c, snr=22, saver_step=50)