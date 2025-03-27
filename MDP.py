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

def compute_dp_matrix(query_diff, target_diff):
    """
    計算初始 DP 矩陣，並應用 pitch interval 調整。

    參數:
        - query_diff: list[int], 查詢歌曲的 MIDI difference
        - target_diff: list[int], 目標歌曲的 MIDI difference
    
    回傳:
        - numpy.ndarray, 調整後的 DP 矩陣
    """
    D = np.abs(np.subtract.outer(query_diff, target_diff))  # 依照你的計算方式
    D_adjusted = apply_pitch_interval_adjustment(D)

    return D_adjusted

def find_best_path(D):
    """
    根據 DP 矩陣尋找最佳路徑：
    1. 從 (0,0) 開始，每次選擇 min(M(i, j)-2, M(i, j+1), M(i+1, j))
    2. 優先級為 M(i, j)-2 > M(i, j+1) > M(i+1, j)
    3. 持續移動，直到達到 target 或 query 的邊界

    參數:
        - D: numpy.ndarray, DP 矩陣

    回傳:
        - list[tuple]: 最佳匹配路徑 (i, j) 座標列表
    """
    i, j = 0, 0
    rows, cols = D.shape
    path = []

    while i < rows or j < cols :

        # 碰到邊界跳出
        if i == rows -1 or j == cols -1:
            path.append((i, j))
            break

        current_val = D[i, j]
        diag_score = current_val - 2
        right_score = D[i, j + 1]
        down_score = D[i + 1, j]
        
        # 确定最小值及优先级
        min_score = min(diag_score, right_score, down_score)
        
        path.append((i, j))          #會跟隨分數計算，可能還需要修改在哪裡append的位置
        if diag_score == min_score:
            # path.append((i, j))
            i += 1
            j += 1
        elif right_score == min_score:
            # path.append((i, j))
            j += 1
        else:
            # path.append((i, j))
            i += 1

    return path

def calculate_matching_score(D_Pitch, D_Beat, Pitch_Path, Beat_Path, w_b=0.8):
    """
    計算音高和節奏的匹配分數
    
    參數:
        D (numpy.ndarray): DP 矩陣
        Pitch_Path (list[tuple]): 最佳匹配路徑 (i, j) 座標列表
        Beat_Path (list[tuple]): 最佳匹配路徑 (i, j) 座標列表
        w_b (float): 節奏權重(預設 0.8)
    
    回傳:
        float: 綜合匹配分數
    """
    
    # 根據 Pitch_Path 抓取 D 中的元素並進行運算
    P = []
    current_row = -1
    row_sum = 0
    for (i, j) in Pitch_Path:
        if i != current_row:
            if current_row != -1:
                P.append(row_sum)
            current_row = i
            row_sum = D_Pitch[i, j]
        else:
            row_sum += D_Pitch[i, j]
    P.append(row_sum)  # 添加最後一行的和

    B = []
    current_row = -1
    row_sum = 0
    for (i, j) in Beat_Path:
        if i != current_row:
            if current_row != -1:
                B.append(row_sum)
            current_row = i
            row_sum = D_Beat[i, j]
        else:
            row_sum += D_Beat[i, j]
    B.append(row_sum)  # 添加最後一行的和

    # 轉換為 numpy array 提高計算效率
    P = np.array(P, dtype=np.float32)
    B = np.array(B, dtype=np.float32)

    print("P = ", P)
    print("B = ", B)
    
    # 計算指數項的總和
    sum_pitch = np.sum(np.exp(-P))
    sum_beat = np.sum(np.exp(-B))
    
    # 加權計算最終分數
    total_score = sum_pitch + w_b * sum_beat
    return total_score



# 測試範例
if __name__ == "__main__":

    # Pitch
    target_diff = [0, 7, 0, 2, 0, -2, -2, 0, -1, 0, -2, 0, -2]
    query_diff = [0, 0, 2, 0, -2, -2, -1, -1, 0, -2, 0, -2]
    # Beat
    td = [2, 2, 2 ,2 ,2 ,3, 1, 2, 2, 2, 2, 2, 3]
    qd = [2, 3, 1, 2, 3, 1, 2, 2, 2, 2, 2, 3]

    D_Pitch = compute_dp_matrix(query_diff, target_diff)
    D_Beat = compute_dp_matrix(qd, td)

    print("D_Pitch = \n", D_Pitch)
    print("D_Beat = \n", D_Beat)

    best_pitch_path = find_best_path(D_Pitch)
    beat_beat_path = find_best_path(D_Beat)
    print("best_pitch_path", best_pitch_path)
    print("beat_beat_path", beat_beat_path)
    
    P_example = [0, 6+0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    B_example = [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print("P_example = ", np.array(P_example, dtype=int))
    print("B_example = ", np.array(B_example, dtype=int))
    
    score = calculate_matching_score(D_Pitch, D_Beat, best_pitch_path, beat_beat_path)
    print(f"Matching Score: {score:.2f}")  # 輸出 Matching Score: 18.77