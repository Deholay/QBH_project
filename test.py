import OnsetDetection
import PitchEstimation
import MelodyMatching
import HiddenMarkovModel

file_path = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/20/20lugo_一閃一閃亮晶晶.wav"
Target_simplified_notation = "115566544332215544332554433211556654433221"

song_name = file_path.split('_')[-1].split('.')[0]

data, rate, time = OnsetDetection.load_audio(file_path)
time_slot_width = 0.01
rho = 0.02
laMbda = 0.7
threshold = 4.5

# ---------------------ONSETDETCTION---------------------
onsets, filtered_signal = OnsetDetection.Onset_Detection(data, rate, time_slot_width, rho, laMbda, threshold)
onsets = OnsetDetection.refine_onsets(onsets, data, rate, time_slot_width, min_interval=0.15, 
                                      start_offset=0.2, end_offset=0.2, PLOT_ONSET = False)

# ---------------------PITCHESTIMATION-------------------
fft_results, pitch_results = PitchEstimation.fft_between_onsets(data, rate, onsets, time_slot_width, L = 10, PLOT_PITCH = False)

beats = PitchEstimation.calculate_beat_intervals(onsets, time_slot_width)
query_midi, query_diff = PitchEstimation.calculate_midi_differences(pitch_results)

# ---------------------MELODYMATCHING--------------------
target_midi, target_diff = MelodyMatching.simplified_notation_to_midi(Target_simplified_notation)
edit_distance, D_matrix = MelodyMatching.calculate_edit_distance(query_diff, target_diff, d = 5)

print(query_diff, "/", target_diff)


# ---------------------HiddenMarkovModel-----------------
states, transition_matrix, state_index = HiddenMarkovModel.build_markov_model(target_diff, min_prob=0.05)
score = HiddenMarkovModel.calculate_score(query_diff, transition_matrix, state_index)

print("Edit Distance:", edit_distance)
print("Dynamic Programming Matrix (D):\n", D_matrix)
#Run Humming data and compare to Target tempo
#Find Edit Distance least ones

print("HMM score: ", score)
#Fix: Some difference is not in Markov Model 方法很笨，直接使用min_prob會有問題，待解決
#Needs: Some Proposed Algorithm