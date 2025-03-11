import numpy as np

def apply_diffusion(transition_counts, diffusion_matrix):
    """
    對 transition_counts 應用 diffusion kernel。
    
    參數:
        - transition_counts: numpy.ndarray, 原始轉移次數矩陣
        - diffusion_matrix: numpy.ndarray, 擴散矩陣 (必須為方陣)
    
    回傳:
        - numpy.ndarray, 經過擴散處理的轉移次數矩陣
    """
    n = transition_counts.shape[0]
    k = diffusion_matrix.shape[0]  # Kernel 大小
    k_half = k // 2  # Kernel 半徑

    new_counts = np.zeros_like(transition_counts, dtype=float)

    for i in range(n):
        for j in range(n):
            if transition_counts[i, j] > 0:
                # 套用 diffusion matrix (需處理邊界情況)
                for di in range(-k_half, k_half + 1):
                    for dj in range(-k_half, k_half + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            new_counts[ni, nj] += transition_counts[i, j] * diffusion_matrix[di + k_half, dj + k_half]
    
    return new_counts

def build_markov_model(target_diff, min_prob=0.001, state_range=(-11, 11)):
    """
    建立 Markov Model 的轉移機率矩陣，應用 diffusion kernel, 並確保 state 範圍涵蓋指定區間。
    
    參數:
        - target_diff: list[int]，目標歌曲的 MIDI difference
        - min_prob: float, 當轉移機率為 0 時的最小值
        - state_range: tuple(int, int), (最小狀態, 最大狀態)
    
    回傳:
        - states: list[int]，所有 MIDI 狀態 (從小到大排序)
        - transition_matrix: numpy.ndarray, 轉移機率矩陣
        - state_index: dict, MIDI Number -> Matrix Index 的映射
    """
    # 確保 state 範圍完整
    states = list(range(state_range[0], state_range[1] + 1))
    state_index = {midi: i for i, midi in enumerate(states)}
    n = len(states)

    # 初始化轉移次數矩陣
    transition_counts = np.zeros((n, n))

    # 計算轉移次數 C(s' | s)
    for i in range(len(target_diff) - 1):
        s, s_next = target_diff[i], target_diff[i + 1]
        if s in state_index and s_next in state_index:
            transition_counts[state_index[s_next], state_index[s]] += 1

    # 定義 diffusion matrix（根據附圖）
    diffusion_matrix = np.array([
        [0.0062, 0.0166, 0.0332, 0.0166, 0.0062],
        [0.0166, 0.0443, 0.0886, 0.0443, 0.0166],
        [0.0332, 0.0886, 0.1773, 0.0886, 0.0322],
        [0.0166, 0.0443, 0.0886, 0.0443, 0.0166],
        [0.0062, 0.0166, 0.0332, 0.0166, 0.0062]
    ])

    # 套用 diffusion kernel
    transition_matrix = apply_diffusion(transition_counts, diffusion_matrix)

    # 設置最小機率 P_m
    transition_matrix[transition_matrix == 0] = min_prob

    return states, transition_matrix, state_index


def get_transition_probability(transition_matrix, state_index, s, s_next, min_prob=0.001):
    """
    :param transition_matrix: 已建構的轉移機率矩陣
    :param state_index: 狀態索引對應表
    :param s: 當前狀態 (MIDI diff)
    :param s_next: 下一個狀態 (MIDI diff)
    :param min_prob: 預設最小機率
    :return: P(s' | s)
    """
    if s not in state_index or s_next not in state_index:
        return min_prob  # 若狀態不在模型中，則直接回傳 min_prob
    i, j = state_index[s], state_index[s_next]
    return transition_matrix[i, j]

def calculate_score(query_diff, transition_matrix, state_index, min_prob=0.001):
    """
    根據 HMM 計算 query 的匹配分數 score_H
    """
    score = 1.0
    for i in range(len(query_diff) - 1):
        s, s_next = query_diff[i], query_diff[i+1]
        score *= get_transition_probability(transition_matrix, state_index, s, s_next, min_prob)
    
    return score



# # 設定 target MIDI 數據，建立 HMM
# target_midi_numbers = [60, 62, 64, 60, 62, 67, 69, 67, 64, 60]
# states, transition_matrix, state_index = build_markov_model(target_midi_numbers)
# # print(state_index[60])


# # 設定 query MIDI 數據
# query_diff = [60, 62, 67, 64]

# # 計算 score_H
# score = calculate_score(query_diff, transition_matrix, state_index)
# # print("Score_H:", score)
