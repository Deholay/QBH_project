import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import convolve

def create_smoother(L):
    # Generate the smoother array based on L
    smoother = np.array([(L - abs(m)) / L**2 if abs(m) < L else 0 for m in range(-L, L+1)])
    return smoother

def fft_between_onsets(data, rate, onsets, time_slot_width, L=10):
    # Generate the smoother based on L
    smoother = create_smoother(L)
    
    # List to store FFT results and detected pitches for each segment
    fft_results = []
    pitch_results = []

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

        # Smooth the FFT result by convolving with the smoother
        smoothed_fft_result = convolve(np.abs(fft_result), smoother, mode='same')

        # Find the fundamental frequency f0 based on conditions (1)-(4)
        max_val = np.max(smoothed_fft_result)
        candidate_m = [
            m for m in range(len(smoothed_fft_result)//2)  # Only consider up to Nyquist frequency
            if smoothed_fft_result[m] > smoothed_fft_result[m+1]  # Condition (1)
            and 80 < freq_axis[m] < rate / 2  # Condition (2)
            and smoothed_fft_result[m] > max_val * 0.2  # Condition (3)
        ]

        # (4) Choose the smallest m that satisfies the conditions
        if candidate_m:
            m0 = min(candidate_m)
            f0 = freq_axis[m0]  # Fundamental frequency
        else:
            f0 = None  # No valid pitch found for this segment

        # Store the pitch and smoothed FFT result for plotting and analysis
        fft_results.append((freq_axis, smoothed_fft_result))
        pitch_results.append(f0)

        # Plot the smoothed FFT result for this segment
        plt.figure(figsize=(10, 5))
        plt.plot(freq_axis[:len(freq_axis)//2], smoothed_fft_result[:len(smoothed_fft_result)//2])
        plt.title(f'Smoothed FFT of Segment between Onset {i} and Onset {i+1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        if f0:
            plt.axvline(x=f0, color='red', linestyle='--', label=f'Pitch f0 = {f0:.2f} Hz')
            plt.legend()
        plt.show()

    return fft_results, pitch_results
