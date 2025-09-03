import MelodyMatching
import HMM
import csv
import numpy as np

def rw_and_build_markov_model(file_path, mode="conv"):
    """
    Load target data and convert the simplified notation to MIDI differences.
    """
    targets_hmms = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()  # Split into song name, simplified notation, and beats
            song_name = parts[0]
            simplified_notation = parts[1].split('/')[0]
            _, target_diff = MelodyMatching.simplified_notation_to_midi(simplified_notation)
            _, hmm_model, _ = HMM.build_markov_model(target_diff, mode, min_prob=0.001, state_range=(-11, 11))
            # targets_hmms[song_name] = np.round(hmm_model, decimals=4)
            targets_hmms[song_name] = hmm_model

    return targets_hmms



if __name__ == "__main__":

    # 執行批量轉換
    target_file = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/Target_tempo_50_utf-8.txt"
    hmms = rw_and_build_markov_model(target_file)

    # 寫入檔案(txt)
    path = 'C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/HMM_midi_diff/HMM_midi_diff.txt'
    file = open(path, 'w', encoding='utf-8')
    for song, model in hmms.items():
        file.write(f"Song Name: {song}\nHMM Model:\n{model}\n")
    file.close()

    # 寫入檔案(csv)
    for song, model in hmms.items():
        song_string_path = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/HMM_midi_diff/" + str(song) + ".csv"
        with open(song_string_path,"w+") as my_csv:
            newarray = csv.writer(my_csv,delimiter=',')
            newarray.writerows(model)