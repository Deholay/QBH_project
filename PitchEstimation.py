import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import convolve

def create_smoother(L):
    # Generate the smoother array based on L
    smoother = np.array([(L - abs(m)) / L**2 if abs(m) < L else 0 for m in range(-L, L+1)])
    return smoother

def fft_between_onsets(data, rate, onsets, time_slot_width, L=10, PLOT_PITCH = True):
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

        if(PLOT_PITCH == True):
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

# Convert pitch to MIDI and calculate MIDI differences
def calculate_midi_differences(pitch_results):
    # Convert each fundamental frequency to a MIDI note number
    midi_numbers = [48 + 12 * np.log2(f0 / 261.63) for f0 in pitch_results if f0]
    
    # Calculate the differences between consecutive MIDI numbers
    midi_differences = np.diff(midi_numbers)

    return midi_numbers, midi_differences

# Calculate beat intervals
def calculate_beat_intervals(onsets, time_slot_width):
    # Calculate intervals between consecutive onsets
    intervals = np.diff(onsets) * time_slot_width  # Convert onset intervals to seconds
    
    # Find the median interval (b0)
    b0 = np.median(intervals)
    
    # Calculate beat for each interval
    beats = [2 ** np.round(np.log2(interval / b0)) for interval in intervals]
    
    return beats


# def plot_pitch(fft_results, pitch_results):
#     num_segments = len(fft_results)
#     fig, axes = plt.subplots(num_segments, 1, figsize=(10, 5 * num_segments))

#     # Plot each segment's FFT result in a separate subplot
#     for i, (freq_axis, smoothed_fft_result) in enumerate(fft_results):
#         ax = axes[i] if num_segments > 1 else axes  # Handle single subplot case
#         ax.plot(freq_axis[:len(freq_axis)//2], smoothed_fft_result[:len(smoothed_fft_result)//2])
#         ax.set_title(f'Smoothed FFT of Segment between Onset {i} and Onset {i+1}')
#         # ax.set_xlabel('Frequency (Hz)')
#         # ax.set_ylabel('Magnitude')
        
#         # Plot fundamental frequency if detected
#         f0 = pitch_results[i]
#         if f0:
#             ax.axvline(x=f0, color='red', linestyle='--', label=f'Pitch f0 = {f0:.2f} Hz')
#             ax.legend()

#     # Adjust layout for better spacing
#     plt.tight_layout()
#     plt.show()