import os

from scipy.fft import fft
import numpy as np
from tqdm import tqdm


def apply_fft_to_database(input_file, output_file='../data/database_1_fft.npz'):
    input_f = np.load(input_file)
    ecg_data = input_f['ecg']
    fourier = []
    for ecg_signal in tqdm(ecg_data):
        fourier.append(np.abs(fft(ecg_signal)))

    np.savez_compressed(output_file,
                        label=input_f['labels'],
                        ecg=ecg_data,
                        fourier=fourier)


apply_fft_to_database('../data/database_1_processed.npz')
