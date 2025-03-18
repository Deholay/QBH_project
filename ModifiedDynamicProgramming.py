import numpy as np

def apply_pitch_interval_adjustment(D):
    """
    對 DP 矩陣 D 進行 pitch interval 調整：
    - 只對 D(i, j) > 6 的元素應用變換。
    
    參數:
        - D: numpy.ndarray, 原始 DP 矩陣

    回傳:
        - numpy.ndarray, 調整後的 DP 矩陣
    """
    D_adjusted = D.copy()

    mask = D > 6  # 找出所有需要調整的 entry
    original_values = D[mask]

    a = (original_values + 5) % 12
    b = (original_values + 5 - a) / 12
    D_adjusted[mask] = np.abs(a - 5) + b

    return D_adjusted

def compute_dp_matrix(target_diff, query_diff):
    """
    計算初始 DP 矩陣，並應用 pitch interval 調整。

    參數:
        - target_diff: list[int], 目標歌曲的 MIDI difference
        - query_diff: list[int], 查詢歌曲的 MIDI difference
    
    回傳:
        - numpy.ndarray, 調整後的 DP 矩陣
    """
    D = np.abs(np.subtract.outer(query_diff, target_diff))
    D_adjusted = apply_pitch_interval_adjustment(D)
    return D_adjusted

# 測試數據
target_diff = [0,7,0,2,0,-2,-2,0,-1,0,-2,0,-2]
query_diff = [0,0,2,0,-2,-2,-1,-1,0,-2,0,-2]

D_matrix = compute_dp_matrix(target_diff, query_diff)
print(D_matrix)