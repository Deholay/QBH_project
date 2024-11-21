import OnsetDetection
import PitchEstimation
import PrintInfo

# file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/10/lugo_愛你一萬年.wav'
# file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/15/15lugo_自由.wav'
# file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/15/15bow_情非得已.wav'
file_path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/thesis/hummingdata/20/20lugo_挪威的森林.wav'
song_name = file_path.split('_')[-1].split('.')[0]

data, rate, time = OnsetDetection.load_audio(file_path)
time_slot_width = 0.01
rho = 0.02
laMbda = 0.7
threshold = 4.5

# ---------------------ONSETDETCTION---------------------
onsets, filtered_signal = OnsetDetection.Onset_Detection(data, rate, time_slot_width, rho, laMbda, threshold)
onsets = OnsetDetection.refine_onsets(onsets, data, rate, time_slot_width, min_interval=0.15, 
                                      start_offset=0.2, end_offset=0.2, PLOT_ONSET = True)

# ---------------------PITCHESTIMATION-------------------
fft_results, pitch_results = PitchEstimation.fft_between_onsets(data, rate, onsets, time_slot_width, L = 10, PLOT_PITCH = True)

midi_numbers, midi_differences = PitchEstimation.calculate_midi_differences(pitch_results)
beats = PitchEstimation.calculate_beat_intervals(onsets, time_slot_width)


# midi ex. [42, 42, 42, 40, 40, 40, 40, 37, 42, 42, 42, 41, 40, 40, 40, 37
# , 42, 42, 42, 40, 40, 37, 35, 35, 37, 37, 35, 37, 42, 42, 40, 40, 40, 37, 42, 41, 42, 40, 40]

# ---------------------INFO PRINT------------------------

# PrintInfo.print_song_info(song_name, midi_numbers, beats)