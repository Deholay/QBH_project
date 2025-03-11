import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import convolve

def load_audio(file_path):
    # Load the audio file
    rate, data = wav.read(file_path)
    time = np.arange(data.shape[0]) / rate
    # If stereo, take one channel
    if len(data.shape) > 1:
        data = data[:, 0]
    return data, rate, time

def Onset_Detection(data, rate, time_slot_width, rho = 0.02, laMbda = 0.7, threshold = 4.5):

    """ -------------Step1. find envelope amplitude----------------------------------------- """
    n0 = int(time_slot_width * rate)
    envelope_amplitude = np.abs(data[:len(data) // n0 * n0].reshape(-1, n0)).mean(axis=1)

    """ -------------Step2. Reduce the effect of the background noise----------------------- """
    reduced_noise_amplitude = np.maximum(envelope_amplitude - rho, 0)

    """ -------------Step3. Normalize the envelope amplitude-------------------------------- """
    # 與thesis不同
    normalized_amplitude = reduced_noise_amplitude / (0.2 + 0.1*np.mean(reduced_noise_amplitude))

    """ -------------Step4. Take the fractional power of the envelope amplitude------------- """
    fractional_amplitude = normalized_amplitude ** laMbda

    """ -------------Step5. Convolution with the envelope match filter---------------------- """
    match_filter = [3, 3, 4, 4, -1, -1, -2, -2, -2, -2, -2, -2]
    filtered_signal = convolve(fractional_amplitude, match_filter, mode='same')

    """ -------------Step6. Thresholding to find the onsets -> 閾值的選擇問題---------------- """
    onsets = np.where(filtered_signal > threshold)[0]

    return onsets, filtered_signal


def refine_onsets(onsets, data, rate, time_slot_width, min_interval = 0.15, start_offset = 0.2, end_offset = 0.2, PLOT_ONSET = True):
    # Convert 0.2 seconds and 0.15 seconds to samples
    start_limit = int(start_offset / time_slot_width)           # start > 0.2
    end_limit = int((rate - end_offset) / time_slot_width)      # end < L - 0.2
    min_interval_limit = int(min_interval / time_slot_width)    # interval > 0.15

    # Exclude onsets near start and end
    refined_onsets = [onset for onset in onsets if start_limit <= onset <= end_limit]

    # Remove onsets that are too close to each other
    # 偵測到 interval < 0.15 提高閾值還沒做
    final_onsets = []
    for i in range(len(refined_onsets)):
        if i == 0 or (refined_onsets[i] - refined_onsets[i-1] >= min_interval_limit):
            final_onsets.append(refined_onsets[i])

    time = np.arange(data.shape[0]) / rate
    if(PLOT_ONSET):
        plt.figure(figsize = (15, 3))
        plt.plot(time, data)
        for onset in final_onsets:
            onset_time = onset * time_slot_width
            plt.axvline(x=onset_time, color='red', linestyle='--', label='Onset' if onset == onsets[0] else "")
        plt.title('ONSET DETECTION')
        plt.show()

    return np.array(final_onsets)