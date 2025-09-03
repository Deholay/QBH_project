import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import librosa # 用於載入音訊和 STFT 等

import FeaturesExtractFormAudio

# 假設您已經有了上一階段實作的特徵提取模組
# from your_feature_extraction_module import extract_features_from_audio
# 為了範例的完整性，這裡會包含一個簡化的 extract_features_from_audio 佔位符
# 您應該替換成您完整的 qbh_feature_extraction_module_v2.py 中的函數

# --- 資料準備 ---
def load_and_preprocess_data(audio_dir, label_dir, max_files=None):
    """
    載入音訊檔案，提取特徵，並生成對應的標籤。
    這是一個高度依賴您資料集結構的函數，您需要根據實際情況修改。

    Args:
        audio_dir (str): 包含 .wav 音訊檔案的目錄。
        label_dir (str): 包含對應標籤檔案的目錄 (例如，每行是 onset 時間的 .txt 檔)。
        max_files (int, optional): 最多處理的檔案數量，用於快速測試。

    Returns:
        tuple: (X, y) 其中 X 是特徵列表 (每個元素是 [k, 204]), y 是標籤列表 (每個元素是 [k, 1])
    """
    all_features = []
    all_labels = []
    file_count = 0

    print(f"開始從 {audio_dir} 載入資料...")
    for filename in sorted(os.listdir(audio_dir)): # 排序以保證順序
        if filename.endswith(".wav"):
            if max_files is not None and file_count >= max_files:
                break
            
            audio_path = os.path.join(audio_dir, filename)
            # 假設標籤檔案與音訊檔案同名，但副檔名為 .txt
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)

            print(f"處理檔案 ({file_count + 1}): {filename}")
            
            # 1. 提取特徵
            #    注意：這裡調用的是佔位符，您需要替換成真實的特徵提取函數
            features_k_204 = FeaturesExtractFormAudio.extract_features_from_audio(audio_path, 
                                                         sr=SAMPLE_RATE, 
                                                         frame_duration_sec=TIME_DOWNSAMPLE_FRAME_SEC,
                                                         hop_length_stft=STFT_HOP_LENGTH)
            
            if features_k_204 is None or features_k_204.shape[0] == 0:
                print(f"  跳過 {filename}：特徵提取失敗或結果為空。")
                continue

            num_k_frames = features_k_204.shape[0]
            
            # 2. 生成標籤 (非常關鍵的一步，需要您根據資料集實現)
            #    假設標籤檔案每行是一個 onset 時間 (秒)
            #    我們需要將這些時間轉換為對應到每個 0.05 秒特徵幀的二元標籤 (0 或 1)
            labels_k = np.zeros((num_k_frames, 1), dtype=np.float32) # 預設所有幀為 non-onset

            if not os.path.exists(label_path):
                print(f"  警告：找不到標籤檔案 {label_path} for {filename}。所有幀將標記為 non-onset。")
            else:
                try:
                    with open(label_path, 'r') as f:
                        onset_times_sec = [float(line.strip()) for line in f if line.strip()]
                    
                    # 將 onset 時間轉換為特徵幀的索引
                    for onset_sec in onset_times_sec:
                        # TIME_DOWNSAMPLE_FRAME_SEC 是每個特徵幀的時長
                        onset_frame_k_idx = int(onset_sec / TIME_DOWNSAMPLE_FRAME_SEC)
                        if 0 <= onset_frame_k_idx < num_k_frames:
                            labels_k[onset_frame_k_idx, 0] = 1.0
                            # 論文提到 "dilate the ground truth and mark the time frames
                            # before and after the original onset frame as the onset frames"
                            # 這裡可以加入擴展邏輯，例如將 onset_frame_k_idx 前後一幀也標為 1
                            if onset_frame_k_idx > 0:
                                labels_k[onset_frame_k_idx - 1, 0] = 1.0 # 前一幀
                            if onset_frame_k_idx < num_k_frames - 1:
                                labels_k[onset_frame_k_idx + 1, 0] = 1.0 # 後一幀
                        # else:
                            # print(f"  警告: Onset time {onset_sec}s 超出特徵幀範圍 for {filename}")
                except Exception as e:
                    print(f"  錯誤：讀取或處理標籤檔案 {label_path} 失敗: {e}")

            all_features.append(features_k_204)
            all_labels.append(labels_k)
            file_count += 1
            
    if not all_features:
        print("錯誤：未能成功處理任何音訊檔案以提取特徵。")
        return np.array([]), np.array([])

    # 將列表中的所有特徵和標籤沿著第一個維度 (時間幀維度) 拼接起來
    # 這樣每個時間幀的特徵向量都成為一個獨立的樣本
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # CNN 的 Conv1D 期望輸入形狀為 (batch_size, steps, channels)
    # 我們的每個樣本是 204 維特徵，可以視為 steps=204, channels=1
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    print(f"資料載入與預處理完成。總共 {X.shape[0]} 個時間幀樣本。")
    print(f"特徵形狀 X: {X.shape}, 標籤形狀 y: {y.shape}")
    
    # 檢查 Onset 比例
    if y.size > 0:
        onset_ratio = np.sum(y) / y.size
        print(f"Onset 樣本比例: {onset_ratio:.4f}")
        if onset_ratio == 0.0:
            print("警告：訓練資料中沒有任何 Onset 標籤！模型可能無法學習。")
        elif onset_ratio > 0.8 : # 或一個非常高的比例
            print("警告：訓練資料中 Onset 標籤比例異常高，請檢查標籤生成邏輯。")


    return X, y

# --- CNN 模型定義 (基於論文圖 17) ---
def build_cnn_model(input_shape=(204, 1)):
    """
    根據論文圖 17 建構 CNN 模型。
    假設輸入是單一時間幀的 204 維特徵。
    """
    inputs = Input(shape=input_shape)

    # 論文圖 17 的卷積層符號 "204x64" 可能指輸入特徵維度204，輸出64個濾波器
    # 1D 卷積的 kernel_size 需要設定，這裡假設為 3
    
    # 區塊 1
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x) # 論文未提，但常規操作
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x) # 假設有池化層
    x = Dropout(0.25)(x) # 論文未提，但常規操作

    # 區塊 2
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x) # 假設有池化層
    x = Dropout(0.25)(x)

    # 分類器
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x) # 論文圖中未明確顯示全連接層大小
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x) # 二元分類 (Onset/Non-Onset)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    print("\nCNN 模型結構:")
    model.summary()
    return model

# --- 主執行流程 ---
if __name__ == '__main__':
    # --- 設定參數 ---

    AUDIO_DIR = "hummingdata\20"  # <--- 修改：包含訓練用 .wav 檔案的目錄
    # AUDIO_DIR = "hummingdata\15"
    # AUDIO_DIR = "hummingdata\10"
    LABEL_DIR = "path/to/your/training_labels_txt"  # <--- 修改：包含對應標籤 .txt 檔案的目錄

    MAX_FILES_TO_PROCESS = None # 設為 None 以處理所有檔案，或設為數字以快速測試，例如 10
    
    # 從您的特徵提取模組獲取參數 (確保與 extract_features_from_audio 中的一致)
    SAMPLE_RATE = 8000
    TIME_DOWNSAMPLE_FRAME_SEC = 0.05
    STFT_HOP_LENGTH = int(SAMPLE_RATE * 0.01) # 確保與特徵提取時一致

    MODEL_SAVE_PATH = "onset_cnn_model.keras" # 或 .h5
    EPOCHS = 50  # <--- 訓練輪數，需要調整
    BATCH_SIZE = 64 # <--- 批次大小，需要調整

    # 1. 載入並預處理資料
    #    您需要確保 load_and_preprocess_data 函數能正確處理您的資料和標籤格式
    print("步驟 1: 載入並預處理資料...")
    if not os.path.isdir(AUDIO_DIR) or not os.path.isdir(LABEL_DIR):
        print(f"錯誤：音訊目錄 '{AUDIO_DIR}' 或標籤目錄 '{LABEL_DIR}' 不存在。")
        print("請創建這些目錄並放入相應的訓練資料，或修改程式碼中的路徑。")
        print("此範例將使用隨機生成的模擬資料進行模型結構測試。")
        # 生成模擬資料 (用於測試模型結構，不能用於真實訓練)
        num_dummy_samples = 1000
        X_dummy = np.random.rand(num_dummy_samples, 204, 1).astype(np.float32)
        y_dummy = np.random.randint(0, 2, size=(num_dummy_samples, 1)).astype(np.float32)
        X_train, X_val, y_train, y_val = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)
        print("已生成模擬資料用於模型測試。")
    else:
        X, y = load_and_preprocess_data(AUDIO_DIR, LABEL_DIR, max_files=MAX_FILES_TO_PROCESS)
        if X.size == 0 or y.size == 0:
            print("錯誤：未能從指定路徑載入任何有效的訓練資料。請檢查路徑和資料。")
            exit()
        # 分割訓練集和驗證集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y if np.sum(y) > 1 else None, random_state=42)
        print(f"訓練集大小: {X_train.shape[0]}, 驗證集大小: {X_val.shape[0]}")


    # 2. 建構 CNN 模型
    print("\n步驟 2: 建構 CNN 模型...")
    model = build_cnn_model(input_shape=(204, 1))

    # 3. 訓練模型
    print("\n步驟 3: 開始訓練模型...")
    if X_train.shape[0] > 0 : # 確保有訓練資料
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
            ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=1)
        
        print("\n模型訓練完成。最佳模型已儲存到:", MODEL_SAVE_PATH)

        # (可選) 評估模型在驗證集上的最終表現
        print("\n在驗證集上評估最終模型:")
        loss, accuracy, precision, recall = model.evaluate(X_val, y_val, verbose=0)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7) # 避免除以零
        print(f"  驗證集損失 (Loss): {loss:.4f}")
        print(f"  驗證集準確率 (Accuracy): {accuracy:.4f}")
        print(f"  驗證集精確率 (Precision): {precision:.4f}")
        print(f"  驗證集召回率 (Recall): {recall:.4f}")
        print(f"  驗證集 F1-Score: {f1_score:.4f}")

    else:
        print("沒有可用的訓練資料，跳過模型訓練。")

    # 4. 如何使用訓練好的模型進行預測 (示例)
    # loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    # example_feature_frame = np.random.rand(1, 204, 1) # 假設一個新的特徵幀
    # prediction_probability = loaded_model.predict(example_feature_frame)
    # is_onset = prediction_probability[0][0] > 0.5 # 假設閾值為 0.5
    # print(f"\n示例預測 - Onset 機率: {prediction_probability[0][0]:.4f}, 是否為 Onset: {is_onset}")

