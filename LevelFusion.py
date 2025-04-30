import OnsetDetection
import PitchEstimation
import MelodyMatching
import HMM
import MDP
import os

# Load the target data
def load_target_data(file_path):
    """
    Load target data and convert the simplified notation to MIDI differences.
    """
    targets = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()  # Split into song name, simplified notation, and beats
            song_name = parts[0]
            simplified_notation = parts[1].split('/')[0]
            _, target_diff = MelodyMatching.simplified_notation_to_midi(simplified_notation)
            targets[song_name] = target_diff
    return targets

# Extract MIDI differences from audio files
def process_audio_to_query_diff(audio_file):
    """
    Process audio to extract MIDI differences.
    """
    time_slot_width = 0.01
    rho = 0.02
    laMbda = 0.7
    threshold = 4.5

    data, rate, _ = OnsetDetection.load_audio(audio_file)
    onsets, _ = OnsetDetection.Onset_Detection(data, rate, time_slot_width, rho, laMbda, threshold)
    onsets = OnsetDetection.refine_onsets(onsets, data, rate, time_slot_width, min_interval=0.15, start_offset=0.2, end_offset=0.2, PLOT_ONSET=False)

    _, pitch_results = PitchEstimation.fft_between_onsets(data, rate, onsets, time_slot_width, PLOT_PITCH=False)
    _, query_diff = PitchEstimation.calculate_midi_differences(pitch_results)
    return query_diff

import numpy as np

def HMM_DP_find_closest_songs(query_path, target_file, transition_matrix_folder):
    """
    Find the closest song for each query audio file by Martching HMM and MDP."
    """
    
    targets = load_target_data(target_file)
    results = []
    correct_count = 0  # Counter for correct matches
    total_count = 0    # Counter for total files
    
    for audio_file in os.listdir(query_path):
        total_count += 1
        if audio_file.endswith('.wav'):  # Adjust file extension as needed
            file_path = os.path.join(query_path, audio_file) # Convert to full path
            query_diff = process_audio_to_query_diff(file_path) 

        # 1. 讀取 HMM_midi_diff 中的 transition_matrix
        transition_matrices = {}
        for filename in os.listdir(transition_matrix_folder):
            if filename.endswith(".csv"):
                song_name = filename[:-4]  # 移除 .csv 副檔名
                filepath = os.path.join(transition_matrix_folder, filename)
                transition_matrices[song_name] =  np.loadtxt(filepath, delimiter=",")
                # print(song_name, transition_matrices[song_name])  # Debug: 查看讀取的 transition matrix

        # 2. 計算所有 target 的 HMM 分數
        hmm_scores = {}
        for song_name, target_diff in targets.items():
            if song_name in transition_matrices:
                transition_matrix = transition_matrices[song_name]
                hmm_scores[song_name] = HMM.calculate_score(query_diff, transition_matrix)
                # print(len(query_diff))  # Debug: 查看 query_diff
                # print(hmm_scores[song_name])  # Debug: 查看 HMM 分數

        # 3. 取 HMM 分數前 20 名
        top_20_targets = sorted(hmm_scores.keys(), key=lambda x: hmm_scores[x], reverse=True)[:120]

        # 正規化 HMM 分數
        top_20_hmm_scores = {song_name: hmm_scores[song_name] for song_name in top_20_targets}
        min_hmm_score = min(top_20_hmm_scores.values())
        max_hmm_score = max(top_20_hmm_scores.values())

        normalized_hmm_scores = {
            song_name: (score - min_hmm_score) / (max_hmm_score - min_hmm_score)
            for song_name, score in top_20_hmm_scores.items()
        }

        # 4. 使用 DP 計算最佳匹配
        best_match = None
        best_score = float('-inf')
        
        for song_name in top_20_targets:
            target_diff = targets[song_name]
            
            # 計算 DP 匹配
            D_Pitch = MDP.compute_dp_matrix(query_diff, target_diff)  
            best_pitch_path = MDP.find_best_path(D_Pitch)
            DP_Score = MDP.calculate_matching_without_beat_score(D_Pitch, best_pitch_path)
            
            # 綜合 HMM 與 DP 分數
            final_score = 0.2 * normalized_hmm_scores[song_name] + 0.8 * DP_Score
            # print(f"Song: {song_name}, HMM Score: {normalized_hmm_scores[song_name]:.4f}, DP Score: {DP_Score:.4f}, Final Score: {final_score:.4f}")

            # 更新最佳匹配
            if final_score > best_score:
                best_score = final_score
                best_match = song_name

        # Check if the predicted best match is correct
        audio_name = os.path.splitext(audio_file)[0].split("_")[1]
        if audio_name == best_match:
            correct_count += 1

        results.append((audio_file, best_match, best_score))

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"Accuracy: {accuracy:.2%}")
    return results


query_path = r"C:\Users\mrjac\Desktop\丁建均老師信號處理專題\QBH_project\hummingdata\20"
target_file = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/Target_tempo_50_utf-8.txt"
transition_matrix_folder="C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/HMM_midi_diff"

results = HMM_DP_find_closest_songs(query_path, target_file, transition_matrix_folder)

for audio_file, best_match, best_score in results:
    print(f"Audio File: {audio_file}, Best Match: {best_match}, Best Score: {best_score}")