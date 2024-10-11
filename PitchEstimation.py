import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

def fft_between_onsets(data, rate, onsets, time_slot_width):
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