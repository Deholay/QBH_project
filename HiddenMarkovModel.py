import numpy as np

def build_markov_model(target_diff, min_prob=0.05):
    """
    建立 Markov Model 的轉移機率矩陣 (不使用 MIDI Diff)。
    參數:
        - target_midi_difference: list[int]，目標歌曲的 MIDI difference
        - min_prob: float, 當轉移機率為 0 時的最小值
    回傳:
        - states: list[int]，所有 MIDI 狀態 (從小到大排序)
        - transition_matrix: numpy.ndarray, 轉移機率矩陣
        - state_index: dict, MIDI Number -> Matrix Index 的映射
    """
    states = sorted(set(target_diff))  # 狀態空間
    state_index = {midi: i for i, midi in enumerate(states)}  # MIDI Number 映射到矩陣索引
    n = len(states)
    
    # 初始化轉移次數矩陣
    transition_counts = np.zeros((n, n))

    # 計算轉移次數 C(s' | s)
    for i in range(len(target_diff) - 1):
        s, s_next = target_diff[i], target_diff[i + 1]
        if s in state_index and s_next in state_index:
            transition_counts[state_index[s_next], state_index[s]] += 1

    # 計算轉移機率矩陣
    transition_matrix = np.zeros((n, n))
    for j in range(n):  # j 是前一個狀態 s
        col_sum = np.sum(transition_counts[:, j])  # 計算 Σ C(s' | s)
        if col_sum > 0:
            transition_matrix[:, j] = transition_counts[:, j] / col_sum
        else:
            transition_matrix[:, j] = 0  # 若無轉移數據，設為 0

    # 設置最小機率 P_m
    transition_matrix[transition_matrix == 0] = min_prob

    # 正規化 (每個 column 之和 = 1)
    transition_matrix /= transition_matrix.sum(axis=0, keepdims=True)

    return states, transition_matrix, state_index

def get_transition_probability(transition_matrix, state_index, s, s_next, min_prob=0.05):
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

def calculate_score(query_diff, transition_matrix, state_index, min_prob=0.05):
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
# print(state_index[60])


# # 設定 query MIDI 數據
# query_diff = [60, 62, 67, 64]

# # 計算 score_H
# score = calculate_score(query_diff, transition_matrix, state_index)
# print("Score_H:", score)
