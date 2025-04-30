import OnsetDetection
import PitchEstimation
import MelodyMatching
import DP
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
            target_beat = [int(digit) for digit in parts[1].split('/')[1]]
            _, target_diff = MelodyMatching.simplified_notation_to_midi(simplified_notation)
            targets[song_name] = [target_diff, target_beat]
    return targets

# Extract MIDI differences from audio files
def process_audio_to_query_diff(audio_file):
    """
    Process audio to extract MIDI differences.
    """
    
    time_slot_width = 0.01
    rho = 0.02
    laMbda = 0.7

    data, rate, _ = OnsetDetection.load_audio(audio_file)
    onsets, _ = OnsetDetection.Onset_Detection(data, rate, time_slot_width, rho, laMbda, k=3.0, use_median=True)
    onsets = OnsetDetection.refine_onsets(onsets, data, rate, time_slot_width, min_interval=0.15, start_offset=0.2, end_offset=0.2, PLOT_ONSET = False)

    _, pitch_results = PitchEstimation.fft_between_onsets(data, rate, onsets, time_slot_width, PLOT_PITCH=False)
    _, query_diff = PitchEstimation.calculate_midi_differences(pitch_results)
    query_beat = PitchEstimation.calculate_beat_intervals(onsets, time_slot_width)
    return query_diff, query_beat

def find_closest_song(query_dir, target_file):
    """
    Find the closest song for each query audio file by calculating edit distance.
    Calculate the accuracy based on correct matches.
    """
    targets = load_target_data(target_file)
    results = []
    correct_count = 0  # Counter for correct matches
    total_count = 0    # Counter for total files

    for audio_file in os.listdir(query_dir):
        if audio_file.endswith('.wav'):  # Adjust file extension as needed
            file_path = os.path.join(query_dir, audio_file) # Convert to full path

            print(f"Processing {audio_file}...")
            query_diff, query_beat = process_audio_to_query_diff(file_path) 

            best_match = None
            best_distance = float('inf')

            for song_name, target_data in targets.items():
                target_diff = target_data[0]
                target_beat = target_data[1]
                distanceD, D = DP.calculate_edit_distance(query_diff, target_diff, d=3)  # Cost 3 is the best
                distanceB, D = DP.calculate_edit_distance(query_beat, target_beat, d=3)  # Cost 3 is the best

                distance = 0.9 * distanceD + 0.1 * distanceB  # Combine distances
                if distance < best_distance:
                    best_distance = distance
                    best_match = song_name

            # Check if the predicted best match is correct
            audio_name = os.path.splitext(audio_file)[0].split("_")[1]
            if audio_name == best_match:
                correct_count += 1

            results.append((audio_file, best_match, best_distance))
            total_count += 1

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0

    print("--------------------------Now, Final Results in", query_dir, "is--------------------------")
    print(f"Accuracy: {accuracy:.2%}")
    return results


# query_dir = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/10"
# query_dir = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/15"
query_dir = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/20"
target_file = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/Target_tempo_50_utf-8.txt"

results = find_closest_song(query_dir, target_file)

for audio_file, best_match, distance in results:
    print(f"Audio File: {audio_file}, Best Match: {best_match}, Edit Distance: {distance}")