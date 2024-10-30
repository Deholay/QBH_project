import matplotlib.pyplot as plt
import numpy as np

import OnsetDetection
import PitchEstimation

# file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/10/lugo_愛你一萬年.wav'
file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/15/15lugo_自由.wav'
# file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/15/15bow_情非得已.wav'
# file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/20/20lugo_挪威的森林.wav'

data, rate = OnsetDetection.load_audio(file_path)
time_slot_width = 0.01
rho = 0.02
laMbda = 0.7
threshold = 4.5
time = np.arange(data.shape[0]) / rate

# ---------------------ONSETDETCTION---------------------
onsets, filtered_signal = OnsetDetection.Onset_Detection(data, rate, time_slot_width, rho, laMbda, threshold)
onsets = OnsetDetection.refine_onsets(onsets, rate, time_slot_width, min_interval=0.15, start_offset=0.2, end_offset=0.2)

OnsetDetection.plot_onsets(time, data, onsets, time_slot_width)

# ---------------------PITCHESTIMATION-------------------
fft_results = PitchEstimation.fft_between_onsets(data, rate, onsets, time_slot_width, L = 10)