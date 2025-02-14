import OnsetDetection
import PitchEstimation
import MelodyMatching

file_path = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/20/20lugo_一閃一閃亮晶晶.wav"
simplified_notation = "11235535567HH5HH6536532123565321"

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
fft_results, pitch_results = PitchEstimation.fft_between_onsets(data, rate, onsets, time_slot_width, L = 10, PLOT_PITCH = False)

beats = PitchEstimation.calculate_beat_intervals(onsets, time_slot_width)
query_midi, query_diff = PitchEstimation.calculate_midi_differences(pitch_results)

# ---------------------EDITDISTANCE----------------------

target_midi, target_diff = MelodyMatching.simplified_notation_to_midi(simplified_notation)
edit_distance, D_matrix = MelodyMatching.calculate_edit_distance(query_diff, target_diff, d = 5)

print("Edit Distance:", edit_distance)
print("Dynamic Programming Matrix (D):\n", D_matrix)

#Run Humming data and compare to Target tempo
#Find Edit Distance least ones