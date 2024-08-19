import tensorflow as tf
import numpy as np
from keras.models import load_model
from AutoencoderModel import NormalizationNoise
from keras.datasets import cifar10
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from matplotlib import pyplot as plt
import pandas as pd

# Load CIFAR-10 dataset
(trainX, _), (testX, _) = cifar10.load_data()

# Normalizing the training and test data
def normalize_pixels(train_data, test_data):
    train_norm = train_data.astype('float32')
    test_norm = test_data.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

x_train, x_test = normalize_pixels(trainX, testX)

compression_ratios = [0.06, 0.09, 0.17]
SNR_train_list = [0, 10, 20, 22]
SNR_test_list = [0, 5, 10, 15, 20, 25]

# Evaluate the model with a single SNR_train and SNR_test pair
def EvaluateModel_single(x_test, comp_ratio, snr_train, snr_test):
    tf.keras.backend.clear_session()
    path = f'/content/checkpoints/CompRatio{comp_ratio}_SNR{snr_train}/Autoencoder.keras'
    autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
    K.set_value(autoencoder.get_layer('normalization_noise').snr_db, snr_test)
    pred_images = autoencoder.predict(x_test) * 255
    pred_images = pred_images.astype('uint8')
    ssim = structural_similarity(testX, pred_images, win_size=3, multichannel=True, channel_axis=-1)
    psnr = peak_signal_noise_ratio(testX, pred_images)
    return psnr

# Plotting function for PSNR vs SNR for multiple SNR_train values
def plot_psnr_vs_snr(x_test, compression_ratio, snr_train_list, snr_test_list, title, x_label, y_label):
    markers = ["*", "s", "o", "X"]
    psnr_values = {}

    table_data = {'SNR_test (dB)': snr_test_list}

    for snr_train in snr_train_list:
        psnr_values[snr_train] = []
        for snr_test in snr_test_list:
            print(f'\n----> Now Getting Data for SNR_train {snr_train} dB and SNR_test {snr_test} dB <----')
            psnr = EvaluateModel_single(x_test, compression_ratio, snr_train, snr_test)
            psnr_values[snr_train].append(psnr)
        
        label = 'ADAPTIVE SNR' if snr_train == 22 else f'Deep JSCC (SNR_train={snr_train}dB)'
        plt.plot(snr_test_list, psnr_values[snr_train], ls='-', marker=markers[snr_train_list.index(snr_train)], label=label)

        # Add PSNR values to the table data
        table_data[label] = psnr_values[snr_train]

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print the PSNR values in a table format
    psnr_df = pd.DataFrame(table_data)
    print("\nPSNR Values for Each Marked Point:")
    print(psnr_df.to_string(index=False))

# Example usage
compression_ratio = 0.06
plot_psnr_vs_snr(x_test, compression_ratio, SNR_train_list, SNR_test_list, title='AWGN channel (k/n=1/16)', x_label='SNR_test (dB)', y_label='PSNR (dB)')
