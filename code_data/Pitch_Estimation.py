import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.fft import fft

def load_audio(file_path):
    # Load the audio file
    rate, data = wav.read(file_path)
    # If stereo, take one channel
    if len(data.shape) > 1:
        data = data[:, 0]
    return data, rate

def find_envelope_amplitude(data, time_slot_width, rate):
    n0 = int(time_slot_width * rate)
    envelope_amplitude = np.abs(data[:len(data) // n0 * n0].reshape(-1, n0)).mean(axis=1)
    return envelope_amplitude

def fractional_power_amplitude(envelope_amplitude, gamma=0.7):
    return envelope_amplitude ** gamma

def match_filter_response():
    return np.array([3, 3, 4, 4, -1, -1, -2, -2, -2, -2, -2, -2])

def refine_onsets(onsets, rate, time_slot_width, min_interval=0.15, start_offset=0.2, end_offset=0.2, data_length=None):
    # Convert 0.2 seconds and 0.15 seconds to samples
    start_limit = int(start_offset / time_slot_width)
    end_limit = int((data_length / rate - end_offset) / time_slot_width)
    min_interval_samples = int(min_interval / time_slot_width)

    # Exclude onsets near start and end
    refined_onsets = [onset for onset in onsets if start_limit <= onset <= end_limit]

    # Remove onsets that are too close to each other
    final_onsets = []
    for i in range(len(refined_onsets)):
        if i == 0 or (refined_onsets[i] - refined_onsets[i-1] >= min_interval_samples):
            final_onsets.append(refined_onsets[i])

    return np.array(final_onsets)

def onset_detection(data, rate, time_slot_width=0.01, gamma=0.7, threshold=1.5):
    # Step 1: Find envelope amplitude
    envelope_amplitude = find_envelope_amplitude(data, time_slot_width, rate)

    # Step 2: Reduce the effect of the background noise
    noise_level = 0.02
    envelope_amplitude = np.maximum(envelope_amplitude - noise_level, 0)

    # Step 3: Normalize the envelope amplitude
    envelope_amplitude = envelope_amplitude / np.mean(envelope_amplitude)

    # Step 4: Take the fractional power of the envelope amplitude
    fractional_amplitude = fractional_power_amplitude(envelope_amplitude, gamma)

    # Step 5: Convolution with the envelope match filter
    match_filter = match_filter_response()
    filtered_signal = convolve(fractional_amplitude, match_filter, mode='same')

    # Step 6: Thresholding to find the onsets
    onsets = np.where(filtered_signal > threshold)[0]

    # Refine onsets: apply exclusion of onsets near start/end and too close intervals
    refined_onsets = refine_onsets(onsets, rate, time_slot_width, data_length=len(data))

    return envelope_amplitude, fractional_amplitude, filtered_signal, refined_onsets

# New function to perform FFT on segments between onsets
def fft_between_onsets(data, onsets, rate, time_slot_width):
    # List to store FFT results for each segment
    fft_results = []
    
    for i in range(len(onsets) - 1):
        # Get the start and end indices for the segment between onset[i] and onset[i+1]
        start_sample = int(onsets[i] * time_slot_width * rate)
        end_sample = int(onsets[i+1] * time_slot_width * rate)
        
        # Extract the segment of the signal
        segment = data[start_sample:end_sample]

        # Perform FFT on the segment
        fft_result = fft(segment)

        # Compute the frequency axis for this segment
        freq_axis = np.fft.fftfreq(len(segment), 1/rate)
        
        # Append the result as (frequencies, magnitudes)
        fft_results.append((freq_axis, np.abs(fft_result)))
        
        # Plot the FFT result for this segment
        plt.figure(figsize=(10, 5))
        plt.plot(freq_axis[:len(freq_axis)//2], np.abs(fft_result)[:len(fft_result)//2])
        plt.title(f'FFT of Segment between Onset {i} and Onset {i+1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()

    return fft_results

# Function to plot each step's result
def plot_results(data, rate, envelope_amplitude, fractional_amplitude, filtered_signal, onsets, time_slot_width):
    time = np.arange(data.shape[0]) / rate
    time_slots = np.arange(envelope_amplitude.shape[0]) * time_slot_width

    plt.figure(figsize=(12, 10))

    # Original signal
    plt.subplot(5, 1, 1)
    plt.plot(time, data)
    plt.title('Original Signal')

    # Envelope amplitude
    plt.subplot(5, 1, 2)
    plt.plot(time_slots, envelope_amplitude)
    plt.title('Envelope Amplitude (Step 1)')

    # Fractional power amplitude
    plt.subplot(5, 1, 3)
    plt.plot(time_slots, fractional_amplitude)
    plt.title('Fractional Power Amplitude (Steps 2, 3, 4)')

    # Convolution result (Step 5)
    plt.subplot(5, 1, 4)
    plt.plot(time_slots, filtered_signal)
    plt.title('Convolution with Envelope Match Filter (Step 5)')

    # Onset detection (Step 6)
    plt.subplot(5, 1, 5)
    plt.plot(time, data)
    for onset in onsets:
        onset_time = onset * time_slot_width
        plt.axvline(x=onset_time, color='red', linestyle='--', label='Onset' if onset == onsets[0] else "")
    plt.title('Detected Onsets (Step 6)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Usage example
file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/15/15lugo_自由.wav'
data, rate = load_audio(file_path)
envelope_amplitude_vals, fractional_amplitude, filtered_signal, onsets = onset_detection(data, rate)
plot_results(data, rate, envelope_amplitude_vals, fractional_amplitude, filtered_signal, onsets, time_slot_width=0.01)

# Perform FFT between detected onsets
fft_results = fft_between_onsets(data, onsets, rate, time_slot_width=0.01)
