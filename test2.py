import OnsetDetection
import PitchEstimation
import MelodyMatching
import DP
import HMM # 確保 HMM 模組和 calculate_score 函數存在
import os
import numpy as np

# --- 函數定義 (精簡版) ---

def load_target_data(file_path):
    """
    (精簡版) 載入目標歌曲資料。假設檔案存在且格式大致正確。
    """
    targets = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                parts = line.strip().split()
                song_name = parts[0]
                simplified_notation = parts[1].split('/')[0]
                target_beat_str = parts[1].split('/')[1]
                target_beat = [int(b) for b in target_beat_str] # 基本轉換
                _, target_diff = MelodyMatching.simplified_notation_to_midi(simplified_notation)
                targets[song_name] = [target_diff, target_beat]
            except Exception as e:
                # 保留最低限度的錯誤提示
                print(f"Warning: Skipping target line due to error: {line.strip()} - {e}")
                continue
    return targets

def load_transition_matrices(folder_path):
    """
    (精簡版) 載入 HMM 轉移矩陣。假設路徑存在且檔案可讀。
    """
    transition_matrices = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            song_name = filename[:-4]
            filepath = os.path.join(folder_path, filename)
            try:
                # 確保以浮點數載入
                transition_matrices[song_name] = np.loadtxt(filepath, delimiter=",", dtype=float)
            except Exception as e:
                 print(f"Warning: Could not load transition matrix for '{song_name}'. Error: {e}")
    return transition_matrices

def process_audio_query(audio_file):
    """
    (精簡版) 處理查詢音訊。假設檔案可讀且能偵測到足夠 Onsets。
    """
    time_slot_width = 0.01
    rho = 0.02
    laMbda = 0.7
    k_dynamic_threshold = 3.0
    use_median_threshold = True

    data, rate, _ = OnsetDetection.load_audio(audio_file)
    # 使用動態閾值
    onsets, _ = OnsetDetection.Onset_Detection(data, rate, time_slot_width, rho, laMbda, k=k_dynamic_threshold, use_median=use_median_threshold)
    onsets = OnsetDetection.refine_onsets(onsets, data, rate, time_slot_width, min_interval=0.15, start_offset=0.2, end_offset=0.2, PLOT_ONSET = False)

    # 基本假設：Onset 數量足夠
    if len(onsets) < 2:
         print(f"Warning: Less than 2 onsets in {os.path.basename(audio_file)}, results might be unreliable.")
         return [], [] # 返回空列表

    _, pitch_results = PitchEstimation.fft_between_onsets(data, rate, onsets, time_slot_width, PLOT_PITCH=False)
    # 基本假設：能產生 pitch results
    if not pitch_results:
         return [], []

    _, query_diff = PitchEstimation.calculate_midi_differences(pitch_results)
    query_beat_raw = PitchEstimation.calculate_beat_intervals(onsets, time_slot_width)

    # 簡化 beat 處理 - 可能需要根據您的 target_beat 意義調整
    # 假設 target_beat 代表每個音符的節拍，長度等於音符數
    query_beat = []
    if query_beat_raw: # 如果至少有兩個音符（一個間隔）
        first_beat = int(round(query_beat_raw[0]))
        query_beat = [first_beat] + [int(round(b)) for b in query_beat_raw]
    elif len(onsets) == 2: # 恰好兩個 onset（一個音符）
        # 沒有間隔可計算 beat_raw，這裡設為一個預設值或空值
        # query_beat = [1] # 例如，假設最短的音符節拍為 1
        pass # 保持 query_beat 為空，讓後續比較失敗

    return query_diff, query_beat


def search_single_query(query_path, target_file, transition_matrix_folder,
                        pitch_weight=0.9, beat_weight=0.1, hmm_weight=0.1,
                        d_cost=3, epsilon=1e-9):
    """
    (精簡版) 搜尋單一查詢並回傳排序結果。
    """
    targets = load_target_data(target_file)
    transition_matrices = load_transition_matrices(transition_matrix_folder)
    query_diff, query_beat = process_audio_query(query_path)

    # 修正後的簡單檢查：檢查 query_diff 長度是否為 0
    if len(query_diff) == 0 and len(query_beat) == 0:
        print(f"Query processing failed for {os.path.basename(query_path)}")
        return [] # 返回空列表

    scores = []
    for song_name, target_data in targets.items():
        target_diff = target_data[0]
        target_beat = target_data[1]

        # 計算編輯距離 (假設 DP.calculate_edit_distance 能處理空列表)
        distanceD, _ = DP.calculate_edit_distance(query_diff, target_diff, d=d_cost)

        # 僅在 query_beat 和 target_beat 長度相符時計算節拍距離 (簡易檢查)
        # 您可以調整這個邏輯，例如長度不符時給固定懲罰
        distanceB, _ = DP.calculate_edit_distance(query_beat, target_beat, d=d_cost)


        # 計算 HMM 分數
        hmm_score_val = 0.0
        if song_name in transition_matrices and len(query_diff) > 0: # 確保 query_diff 非空
            try:
                transition_matrix = transition_matrices[song_name]
                hmm_score_val = HMM.calculate_score(query_diff, transition_matrix)
                hmm_score_val = max(0, hmm_score_val) # 確保非負
            except Exception as e:
                print(f"Warning: Error calculating HMM score for {song_name}: {e}")
                hmm_score_val = 0.0

        # --- 使用您原來的分數組合公式，但加入 epsilon 避免 log(0) ---
        # 注意：這個公式假設 HMM 分數越高越好，但 log(低分) 會給出大的負數，
        # 可能導致低 HMM 分數的歌曲排名意外靠前。
        # 如果 HMM 分數越高越好，分數公式更常見的是減去 HMM 分數或其對數。
        # 這裡暫時保留您原來的公式結構。
        try:
            # 檢查 hmm_score_val + epsilon 是否大於 0
            if hmm_score_val + epsilon <= 0:
                 log_hmm_term = -float('inf') # 或是一個很大的負數作為懲罰
                 print(f"Warning: Cannot compute log for non-positive HMM score + epsilon for {song_name}")
            else:
                 log_hmm_term = np.log(hmm_score_val)

            # 如果 log 計算失敗，給予極大分數
            if np.isinf(log_hmm_term):
                 final_score = float('inf')
            else:
                 final_score = pitch_weight * distanceD + beat_weight * distanceB - hmm_weight * log_hmm_term

        except ValueError as e: # 處理 distanceB/D 可能非數值的情況 (如果 DP 返回特殊值)
            print(f"Warning: Error calculating final score for {song_name}: {e}")
            final_score = float('inf')


        scores.append({'name': song_name, 'score': final_score, 'distanceD' : distanceD, 'distanceB' : distanceB, 'hmm_score': log_hmm_term})
        # -----------------------------------------------------------

    # 根據分數排序 (分數越低越好)
    scores.sort(key=lambda x: float('inf') if np.isnan(x['score']) or np.isinf(x['score']) else x['score'])
    return scores

# --- 主執行區塊 ---
if __name__ == "__main__":
    # --- 設定路徑 ---
    # query_path = r"C:/Path/To/Your/Single/Query/humming.wav" # <--- 請修改為您的單一查詢檔案路徑

    query_path = r"hummingdata\15\15bow_在凌晨.wav"

    # query_path = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/20/20lugo_對面的女孩看過來.wav"
    # query_path = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/20/20lugo_靜止.wav"



    target_file = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/Target_tempo_50_utf-8.txt"
    transition_matrix_folder = r"C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/Project_in_Lin's_thesis/HMM_midi_diff"

    # --- 設定權重和參數 ---
    P_WEIGHT = 0.95
    B_WEIGHT = 0.0125 # 節拍距離的權重
    H_WEIGHT = 0.0375 # HMM 分數的權重
    D_COST = 3
    EPSILON = 1e-9 # 防止 log(0)

    # --- 執行搜尋 ---
    print(f"Starting search for: {os.path.basename(query_path)}")
    ranked_results = search_single_query(
        query_path,
        target_file,
        transition_matrix_folder,
        pitch_weight=P_WEIGHT,
        beat_weight=B_WEIGHT,
        hmm_weight=H_WEIGHT,
        d_cost=D_COST,
        epsilon=EPSILON
    )

    # --- 輸出排名結果 ---
    if ranked_results:
        print("\n--- Search Results ---")
        for rank, result in enumerate(ranked_results, 1):
            print(f"Rank {rank}: {result['name']}, Score: {result['score']:.4f}, DistanceD: {result['distanceD']}, DistanceB: {result['distanceB']}, HMM Score: {result['hmm_score']}")
            # 只顯示前 20 名結果 (可選)
            if rank >= 20:
                print("...")
                break
    else:
        print("\nSearch could not be completed.")