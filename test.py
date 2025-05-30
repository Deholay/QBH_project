import OnsetDetection
import PitchEstimation
import MelodyMatching
import HMM
import DP

file_path = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/20/20lugo_對面的女孩看過來.wav"
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
# target_midi, target_diff = MelodyMatching.simplified_notation_to_midi(Target_simplified_notation)
# edit_distance, D_matrix = DP.calculate_edit_distance(query_diff, target_diff, d = 5)

import main
target_file = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/Target_tempo_50_utf-8.txt"
targets = main.load_target_data(target_file)

for song_name, target_diff in targets.items():
    distance, D = DP.calculate_edit_distance(query_diff, target_diff, d=4)
    print(f"Distance between {song_name} and target : {distance}")


# print(query_diff, "/", target_diff)


# ---------------------HiddenMarkovModel-----------------
# states, transition_matrix, state_index = HMM.build_markov_model(target_diff, min_prob=0.001)
# score = HMM.calculate_score(query_diff, transition_matrix, state_index)

# print("Edit Distance:", edit_distance)
# print("Dynamic Programming Matrix (D):\n", D_matrix)
#Run Humming data and compare to Target tempo
#Find Edit Distance least ones

# print("HMM score: ", score)
# print("states: ", states)
# print("state_index: ", state_index)
# print("matrix: \n", transition_matrix)
#Needs: Some Proposed Algorithm