# -- coding: utf-8 --
"""
Created on Thu Dec 12 20:57:10 2019

@author: ROOT
"""

import tensorflow as tf
import os
import numpy as np
from keras.models import load_model
from AutoencoderModel import NormalizationNoise
from keras.datasets import cifar10
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio 
from matplotlib import pyplot as plt
from tensorflow.keras.losses import mse

#all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
#all_tensors[100:200]
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
compression_ratios = [0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49]
#compression_ratios = [0.06, 0.09, 0.17]
#compression_ratios = [0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49]
SNR_dB = [0, 10, 20]

snr = 20
def EvaluateModel(x_test, compression_ratios, snr, mode='multiple'):
    if mode=='single':
        tf.keras.backend.clear_session()
        comp_ratio=compression_ratios
        path = '/content/checkpoints/CompRatio{0}_SNR{1}/Autoencoder.keras'.format(comp_ratio, snr)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test)*255
        pred_images = pred_images.astype('uint8')
        ssim = structural_similarity(testX, pred_images, win_size=3, multichannel=True, channel_axis=-1)
        psnr = peak_signal_noise_ratio(testX, pred_images)
        return pred_images, psnr, ssim
    elif mode=='multiple':  
        model_dic = {'SNR':[], 'Pred_Images':[], 'PSNR':[], 'SSIM':[]}
        model_dic['SNR'].append(snr)
        for comp_ratio in compression_ratios:
            tf.keras.backend.clear_session()
            path = '/content/checkpoints/CompRatio{0}_SNR{1}/Autoencoder.keras'.format(comp_ratio, snr)
            autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
            K.set_value(autoencoder.get_layer('normalization_noise').snr_db, snr)
            pred_images = autoencoder.predict(x_test)*255
            pred_images = pred_images.astype('uint8')
            ssim = structural_similarity(testX, pred_images, win_size=3, multichannel=True, channel_axis=-1)
            psnr = peak_signal_noise_ratio(testX, pred_images)
            model_dic['Pred_Images'].append(pred_images)
            model_dic['PSNR'].append(psnr)
            model_dic['SSIM'].append(ssim)
        return model_dic
            

def plot_model(x_test, compression_ratios, snr_lst, title, x_lablel, y_label):
    markers = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
    colors = ['#800080', '#FF00FF', '#000080', '#008080', '#00FFFF', '#008000', '#00FF00']
    history = []
    i=0
    for snr in snr_lst:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = EvaluateModel(x_test, compression_ratios, snr, mode='multiple')
        history.append(model_dic)
        label='Deep JSCC (SNR={0}dB)'.format(snr)
        plt.plot(compression_ratios, model_dic['PSNR'], ls = '--', c = colors[i], marker = markers[i], label=label)
        
        # k/n ve PSNR değerlerini yazdırma
        for k, psnr in zip(compression_ratios, model_dic['PSNR']):
          print(f"k/n: {k}, PSNR: {psnr:.2f} dB")
        
        i += 1
        plt.title(title)
        plt.xlabel(x_lablel)
        plt.ylabel(y_label)
        plt.grid(True)
    plt.ylim(10,35)
    plt.show()
    return history

    
snr_lst=[10]
history = plot_model(x_test, compression_ratios, snr_lst, title='AWGN Channel', x_lablel='k/n', y_label='PSNR (dB)')   

#model_dic = EvaluateModel(x_test, compression_ratios, snr, mode='multiple')
#markers = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
#colors = ['#800080', '#FF00FF', '#000080', '#008080', '#00FFFF', '#008000', '#00FF00', '#808000', '#FF0000', '#800000', '#000000', '#000080']
#plt.plot(compression_ratios, model_dic['PSNR'], ls = '--', c = 'm', marker = 'o')
