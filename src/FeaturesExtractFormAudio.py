import numpy as np
import librosa # 使用 librosa 進行 STFT 和音訊處理更方便
from scipy.signal import lfilter, convolve # 用於匹配濾波器和 HPS 的卷積

# --- 常數與參數設定 (部分參數可能需要根據論文詳細內容或實驗調整) ---
# 音訊處理參數
SAMPLE_RATE = 8000
STFT_N_FFT = 512
STFT_HOP_LENGTH = int(SAMPLE_RATE * 0.01) # 80 samples for 10ms
STFT_WIN_LENGTH = int(SAMPLE_RATE * 0.025) # 200 samples for 25ms
# STFT_WINDOW = 'hann' # librosa.stft 預設使用 hann window

# 特徵提取參數 (基於論文 3.5 節)
NUM_OCTAVE_BANDS = 6
FIRST_OCTAVE_START_FREQ = 80.0 # 確保是浮點數
TIME_DOWNSAMPLE_FRAME_SEC = 0.05

# HPS 參數 (論文 3.5.5 節, 公式 21)
HPS_R_MAX = 5 # 諧波疊加次數 (m=1 到 5)

# 匹配濾波器參數 (論文 3.5.6 節)
NUM_MATCHED_FILTERS = 10

# --- 輔助函數 (get_octave_bands_freq_indices, extract_band_features, calculate_variations 同前一版本) ---
def get_octave_bands_freq_indices(n_fft, sr, num_bands, first_band_start_freq):
    """
    計算 STFT 頻率軸上對應各個八度頻帶的索引。
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_indices = []
    current_f_start = float(first_band_start_freq)

    for i in range(num_bands):
        current_f_end = current_f_start * 2.0
        start_idx = np.searchsorted(freqs, current_f_start, side='left')
        end_idx = np.searchsorted(freqs, current_f_end, side='left')
        end_idx = min(end_idx, len(freqs))

        if start_idx < end_idx:
            band_indices.append((start_idx, end_idx))
        else:
            # print(f"警告：無法為第 {i+1} 個八度頻帶形成有效索引 (F_start={current_f_start:.2f}, F_end={current_f_end:.2f})")
            # 如果頻帶不足，後續特徵維度可能不匹配，需要處理
            # 為了簡化，這裡假設總能生成足夠的頻帶，或在主函數中處理維度不匹配
            pass

        current_f_start = current_f_end
        if current_f_start >= sr / 2.0:
            break
    return band_indices, freqs

def extract_band_features(stft_magnitude_or_power, band_indices):
    """
    提取頻帶特徵 (能量總和與最大值)。
    stft_magnitude_or_power: 幅度譜或功率譜 [頻率, 時間]
    """
    num_frames = stft_magnitude_or_power.shape[1]
    num_bands_actual = len(band_indices)

    # 如果 band_indices 為空，表示沒有有效的頻帶被劃分
    if num_bands_actual == 0:
        # 返回空的或充滿零的特徵陣列，維度應與預期一致以便後續拼接
        # 這裡假設如果沒有頻帶，則這些特徵為0
        return np.zeros((num_frames, NUM_OCTAVE_BANDS)), np.zeros((num_frames, NUM_OCTAVE_BANDS))


    # 初始化為預期的頻帶數，如果實際頻帶數較少，未覆蓋的部分將保持為0
    features_sum = np.zeros((num_frames, NUM_OCTAVE_BANDS))
    features_max = np.zeros((num_frames, NUM_OCTAVE_BANDS))

    for n in range(num_bands_actual): # 只遍歷實際生成的頻帶
        start_idx, end_idx = band_indices[n]
        if start_idx >= end_idx : continue # 跳過無效頻帶

        band_data = stft_magnitude_or_power[start_idx:end_idx, :]
        if band_data.size > 0:
            features_sum[:, n] = np.sum(band_data, axis=0)
            features_max[:, n] = np.max(band_data, axis=0)
        # else features_sum/max[:, n] 保持為 0

    return features_sum, features_max


def calculate_variations(feature_k_n):
    """
    計算特徵的時間變化量。
    feature_k_n: 形狀為 [降採樣後的時間幀k, 頻帶數n]
    """
    variations = []
    num_k_frames, num_bands = feature_k_n.shape

    if num_k_frames == 0: # 如果沒有降採樣幀，返回空的變化量
        for _ in range(5): variations.append(np.zeros((0, num_bands)))
        return variations

    # (16) Xssd
    Xssd = np.zeros_like(feature_k_n)
    if num_k_frames > 1: Xssd[1:, :] = feature_k_n[1:, :] - feature_k_n[:-1, :]
    elif num_k_frames == 1: Xssd[0, :] = feature_k_n[0,:] # 或設為0
    variations.append(Xssd)

    # (17) Xssd2
    Xssd2 = np.zeros_like(Xssd)
    if num_k_frames > 1: Xssd2[1:, :] = Xssd[1:, :] - Xssd[:-1, :]
    elif num_k_frames == 1: Xssd2[0,:] = Xssd[0,:]
    variations.append(Xssd2)

    # (18) Xssd101
    Xssd101 = np.zeros_like(feature_k_n)
    if num_k_frames >= 3: Xssd101[1:-1, :] = feature_k_n[2:, :] - feature_k_n[:-2, :]
    elif num_k_frames == 2: Xssd101[0, :] = feature_k_n[1,:] - feature_k_n[0,:] # 近似
    elif num_k_frames == 1: Xssd101[0,:] = feature_k_n[0,:] # 或設為0
    variations.append(Xssd101)

    # (19) Xssd102
    Xssd102 = np.zeros_like(feature_k_n)
    if num_k_frames >= 4: Xssd102[1:-2, :] = feature_k_n[3:, :] - feature_k_n[:-3, :]
    elif num_k_frames == 3: Xssd102[0,:] = feature_k_n[2,:] - feature_k_n[0,:] # 近似
    elif num_k_frames <= 2: pass # 無法計算，保持為0
    variations.append(Xssd102)

    # (20) Xssd201
    Xssd201 = np.zeros_like(feature_k_n)
    if num_k_frames >= 4: 
        Xssd201[2:-1, :] = feature_k_n[3:, :] - feature_k_n[:-3, :] # 論文公式似乎有誤，應為 Xss(k+1) - Xss(k-2)
                                                                                   # k from 2 to K-2. Xss[k_idx+1] - Xss[k_idx-2]
        for k_idx in range(2, num_k_frames - 1):
             Xssd201[k_idx, :] = feature_k_n[k_idx + 1, :] - feature_k_n[k_idx - 2, :]
    elif num_k_frames == 3: # 只有 k_idx = 1 (對應論文的 k=2)
        Xssd201[1,:] = feature_k_n[2,:] - feature_k_n[0,:] # 近似
    elif num_k_frames <=2: pass
    variations.append(Xssd201)

    return variations

def harmonic_product_spectrum(spectrum_frame, sr, n_fft, r_max=5):
    """
    計算單一頻譜幀的諧波乘積頻譜 (HPS) 並估計 F0。
    spectrum_frame: 一幀的幅度譜 (線性幅度)
    r_max: 疊加的諧波數量 (論文中為5)
    返回: 估計的 F0 (Hz)
    """
    if spectrum_frame.ndim != 1 or spectrum_frame.size == 0:
        return 0.0 # 無效輸入

    hps_spectrum = np.copy(spectrum_frame)
    # 確保下採樣時不會超出原始頻譜長度
    # 並且只在有效頻率範圍內操作 (例如，高於某個最低F0，低於某個最高F0)
    min_f0_hz = 30 # 假設最低F0
    max_f0_hz = sr / (2.0 * r_max) # 確保 f/r 不會太高

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    valid_indices = (freqs >= min_f0_hz) & (freqs <= max_f0_hz)
    
    if not np.any(valid_indices):
        return 0.0

    # 對齊hps_spectrum到與valid_indices相同的長度
    if len(hps_spectrum) > len(freqs):
        hps_spectrum = hps_spectrum[:len(freqs)]
    
    # 僅在有效頻率範圍內計算HPS
    hps_result_valid_range = np.copy(hps_spectrum[valid_indices])


    for r in range(2, r_max + 1):
        downsampled_spectrum = spectrum_frame[::r] # 降採樣
        # 確保長度對齊以進行乘法
        len_to_match = len(hps_result_valid_range)
        
        # 從降採樣的頻譜中取出對應有效範圍的部分
        # 計算降採樣後，原始有效頻率索引對應到新索引
        downsampled_freqs = freqs[::r]
        
        # 找到降採樣頻譜中與原始有效頻率範圍重疊的部分
        # 這一步比較複雜，簡化處理：直接截斷或插值
        # 簡化：假設我們只取 downsampled_spectrum 的前 len_to_match 個點
        current_downsampled_segment = downsampled_spectrum[:len_to_match]
        if len(current_downsampled_segment) < len_to_match: # 如果不夠長，補零
            current_downsampled_segment = np.pad(current_downsampled_segment,
                                                 (0, len_to_match - len(current_downsampled_segment)),
                                                 'constant')
        hps_result_valid_range *= current_downsampled_segment

    if hps_result_valid_range.size == 0:
        return 0.0
        
    # 找到 HPS 最大值對應的頻率
    peak_index_in_valid_range = np.argmax(hps_result_valid_range)
    # 將此索引映射回原始頻率軸的有效索引部分
    original_indices_of_valid_range = np.where(valid_indices)[0]
    
    if peak_index_in_valid_range < len(original_indices_of_valid_range):
        f0_index = original_indices_of_valid_range[peak_index_in_valid_range]
        estimated_f0 = freqs[f0_index]
        return estimated_f0
    else: # 應該不會發生，除非hps_result_valid_range為空
        return 0.0


def create_example_matched_filters(num_filters, filter_len=50):
    """
    創建示例匹配濾波器 (佔位符)。
    實際應用中需要根據論文[15]或實驗設計。
    返回: 一個濾波器列表。
    """
    filters = []
    # 示例：簡單的上升/下降/脈衝等形狀
    t = np.linspace(-1, 1, filter_len)
    # 1. 上升沿
    filters.append(np.linspace(0, 1, filter_len))
    # 2. 下降沿
    filters.append(np.linspace(1, 0, filter_len))
    # 3. 三角波
    filters.append(1 - np.abs(t))
    # 4. 高斯脈衝 (近似)
    filters.append(np.exp(-5 * t**2))
    # 5. 符號變化的濾波器 (類似論文 胡哲銘.pdf Fig 7.9)
    mf_paper1 = np.array([1]* (filter_len//3) + [-1]*(filter_len//3) + [0.5]*(filter_len - 2*(filter_len//3)))
    if len(mf_paper1) < filter_len: mf_paper1 = np.pad(mf_paper1, (0, filter_len-len(mf_paper1)))
    filters.append(mf_paper1[:filter_len])

    # 補齊到 num_filters 個
    while len(filters) < num_filters:
        # 可以創建更多變種，或重複使用/稍微修改已有的
        filters.append(np.random.randn(filter_len) * 0.1) # 隨機噪聲作為佔位
    return filters[:num_filters]


# --- 主特徵提取函數 ---
def extract_features_from_audio(audio_path):
    """
    從音訊檔案中提取論文描述的204維特徵。
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            print(f"錯誤：載入的音訊檔案 {audio_path} 為空。")
            return None
    except Exception as e:
        print(f"錯誤：無法載入音訊檔案 {audio_path}: {e}")
        return None

    # 1. 計算 STFT
    D_magnitude = np.abs(librosa.stft(y, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH, win_length=STFT_WIN_LENGTH))
    D_power = D_magnitude**2
    num_original_frames = D_magnitude.shape[1]

    if num_original_frames == 0:
        print(f"錯誤：STFT 結果為空 for {audio_path} (音訊可能太短)。")
        return None

    # 2. 頻帶劃分
    band_indices, freqs = get_octave_bands_freq_indices(STFT_N_FFT, sr, NUM_OCTAVE_BANDS, FIRST_OCTAVE_START_FREQ)
    actual_num_bands = len(band_indices)

    # 3. 提取頻帶基礎特徵 X6a, X6b, Y6a, Y6b
    X6a, X6b = extract_band_features(D_magnitude, band_indices)
    Y6a, Y6b = extract_band_features(D_power, band_indices)

    # 4. 時間降採樣與聚合
    samples_per_downsample_frame_audio = int(sr * TIME_DOWNSAMPLE_FRAME_SEC) # 音訊樣本數
    frames_per_downsample_frame_stft = samples_per_downsample_frame_audio // STFT_HOP_LENGTH # STFT幀數
    
    if frames_per_downsample_frame_stft == 0:
        print(f"警告：TIME_DOWNSAMPLE_FRAME_SEC ({TIME_DOWNSAMPLE_FRAME_SEC}s) 太短，不足以包含一個 STFT 幀的 hop_length。")
        return None
        
    num_downsampled_frames = num_original_frames // frames_per_downsample_frame_stft

    if num_downsampled_frames == 0:
        print(f"警告：音訊長度不足以形成至少一個 {TIME_DOWNSAMPLE_FRAME_SEC} 秒的降採樣幀 for {audio_path}.")
        return None

    # 初始化聚合特徵的字典
    aggregated_X = {}
    aggregated_Y = {}
    
    # 聚合函數
    def aggregate(data_t_n, agg_type='sum'):
        # data_t_n shape: [original_frames, num_bands]
        resampled = np.zeros((num_downsampled_frames, NUM_OCTAVE_BANDS)) #確保輸出維度固定
        for k in range(num_downsampled_frames):
            start_idx = k * frames_per_downsample_frame_stft
            end_idx = (k + 1) * frames_per_downsample_frame_stft
            segment = data_t_n[start_idx:min(end_idx, num_original_frames), :]
            if segment.size > 0:
                if agg_type == 'sum':
                    resampled[k, :segment.shape[1]] = np.sum(segment, axis=0) # 只填充實際頻帶的數據
                elif agg_type == 'max':
                    resampled[k, :segment.shape[1]] = np.max(segment, axis=0)
        return resampled

    aggregated_X['ss'] = aggregate(X6a, 'sum')
    aggregated_X['ms'] = aggregate(X6a, 'max')
    aggregated_X['sm'] = aggregate(X6b, 'sum')
    aggregated_X['mm'] = aggregate(X6b, 'max')

    aggregated_Y['ss'] = aggregate(Y6a, 'sum')
    aggregated_Y['ms'] = aggregate(Y6a, 'max')
    aggregated_Y['sm'] = aggregate(Y6b, 'sum')
    aggregated_Y['mm'] = aggregate(Y6b, 'max')

    # 5. 計算 STFT 相關的變化量特徵 (192 維)
    all_stft_variation_features_list = []
    # X 系列: 4 種類型 * 5 種變化
    for key in ['ss', 'ms', 'sm', 'mm']:
        variations = calculate_variations(aggregated_X[key]) # aggregated_X[key] shape: [k, NUM_OCTAVE_BANDS]
        for var_feat in variations: # var_feat shape: [k, NUM_OCTAVE_BANDS]
            all_stft_variation_features_list.append(var_feat)

    # Y 系列: 4 種類型 * 3 種長期變化
    long_term_var_indices = [2, 3, 4] # 對應 Xssd101, Xssd102, Xssd201
    for key in ['ss', 'ms', 'sm', 'mm']:
        variations = calculate_variations(aggregated_Y[key])
        for idx in long_term_var_indices:
            all_stft_variation_features_list.append(variations[idx])
    
    # 拼接成 [k, 192]
    # 預期 all_stft_variation_features_list 有 32 個元素，每個元素 shape [k, 6]
    if not all_stft_variation_features_list or any(f.shape[0] != num_downsampled_frames for f in all_stft_variation_features_list):
        print(f"錯誤：STFT 變化量特徵計算出錯或長度不一致 for {audio_path}")
        return None
    
    stft_features_192d = np.concatenate(all_stft_variation_features_list, axis=1)
    if stft_features_192d.shape[1] != 192:
        print(f"警告：STFT 特徵維度 ({stft_features_192d.shape[1]}) 不等於預期的 192 for {audio_path}。可能是頻帶數不足或計算錯誤。")
        # 進行填充或返回錯誤
        if stft_features_192d.shape[1] < 192:
            padding = np.zeros((num_downsampled_frames, 192 - stft_features_192d.shape[1]))
            stft_features_192d = np.concatenate((stft_features_192d, padding), axis=1)
        else: # 比192多，不太可能發生，除非 NUM_OCTAVE_BANDS > 6
            stft_features_192d = stft_features_192d[:, :192]


    # 6. 提取基頻相關特徵 (2 維) - 論文 3.5.5
    f0_values_k = np.zeros(num_downsampled_frames)
    for k_idx in range(num_downsampled_frames):
        start_stft_frame = k_idx * frames_per_downsample_frame_stft
        end_stft_frame = (k_idx + 1) * frames_per_downsample_frame_stft
        # 取該大幀內所有 STFT 短幀的平均頻譜 (或中心幀)
        # 簡化：取中心短幀的頻譜
        center_stft_frame_idx = start_stft_frame + frames_per_downsample_frame_stft // 2
        if center_stft_frame_idx < num_original_frames:
            spectrum_for_f0 = D_magnitude[:, center_stft_frame_idx]
            f0_values_k[k_idx] = harmonic_product_spectrum(spectrum_for_f0, sr, STFT_N_FFT, r_max=HPS_R_MAX)
        else:
            f0_values_k[k_idx] = 0 # 或者用前一個值填充

    f0_feature = f0_values_k.reshape(-1, 1)
    f0_variation_feature = np.zeros_like(f0_feature)
    if num_downsampled_frames > 1:
        f0_variation_feature[1:] = f0_feature[1:] - f0_feature[:-1]
    elif num_downsampled_frames == 1:
        f0_variation_feature[0] = f0_feature[0] # 或設為0

    # 7. 提取匹配濾波器特徵 (10 維) - 論文 3.5.6 (簡化版/佔位符)
    matched_filter_bank = create_example_matched_filters(NUM_MATCHED_FILTERS)
    mf_features_k = np.zeros((num_downsampled_frames, NUM_MATCHED_FILTERS))

    # 獲取原始音訊的包絡線 (一個簡化的方法)
    # 您也可以直接在原始音訊 y 上應用濾波器
    # 為了與論文[15]的 "fractional power envelope match filter" 有一點點關聯，我們用包絡
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=STFT_HOP_LENGTH) # 與STFT幀對齊

    for k_idx in range(num_downsampled_frames):
        start_stft_frame = k_idx * frames_per_downsample_frame_stft
        end_stft_frame = (k_idx + 1) * frames_per_downsample_frame_stft
        
        # 對應到 onset_env 的幀索引
        # onset_env 的長度與 STFT 幀數相同
        segment_env = onset_env[start_stft_frame:min(end_stft_frame, len(onset_env))]

        if segment_env.size > 0:
            for filter_idx, mf in enumerate(matched_filter_bank):
                if len(segment_env) >= len(mf): # 確保訊號段長度足夠進行卷積
                    # 這裡的卷積是在已經是特徵序列 (包絡強度) 上進行
                    # 論文中可能是在原始時域訊號上做
                    conv_result = convolve(segment_env, mf, mode='valid') # valid 模式確保不越界
                    if conv_result.size > 0:
                        mf_features_k[k_idx, filter_idx] = np.max(np.abs(conv_result)) # 取最大絕對值作為特徵
                    else:
                        mf_features_k[k_idx, filter_idx] = 0 # 卷積結果為空
                else: # 訊號段太短
                     mf_features_k[k_idx, filter_idx] = 0 # 或其他處理
        # else mf_features_k 保持為0

    # 8. 拼接所有特徵
    final_features = np.concatenate((stft_features_192d, f0_feature, f0_variation_feature, mf_features_k), axis=1)

    if final_features.shape[1] != 204:
        print(f"警告：最終特徵維度 ({final_features.shape[1]}) 不等於預期的 204 for {audio_path}。")
        # 根據需要進行填充或截斷以確保維度一致 (這裡不處理，留給調用者)

    return final_features

# --- 測試代碼 ---
if __name__ == '__main__':
    # --- 請替換為您實際的音訊檔案路徑 ---
    test_audio_file = "C:/Users/mrjac/Desktop/丁建均老師信號處理專題/QBH_project/hummingdata/20/20lugo_愛你一萬年.wav"
    # 確保 SAMPLE_RATE 與您的音訊檔案一致，或者 librosa.load 會自動重採樣

    print(f"正在處理測試音訊: {test_audio_file}")
    features = extract_features_from_audio(test_audio_file)

    if features is not None:
        print(f"成功提取特徵，形狀: {features.shape}")
        # 預期形狀: (降採樣後的幀數 k, 204)
    else:
        print("特徵提取失敗。")

