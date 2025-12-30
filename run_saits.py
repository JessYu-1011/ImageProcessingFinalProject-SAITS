import numpy as np
from pypots.imputation import SAITS
from data_loader import load_and_process_data
from plot import plot_imputation_result

if __name__ == '__main__':
    FILE_PATH = './dataset/weather.csv'
    WINDOW_SIZE = 144

    # 1. Load data
    X, scaler, feature_names = load_and_process_data(FILE_PATH, WINDOW_SIZE)

    # 2. simulate missing data (randomly mask 10% for testing)
    X_input = X.copy()
    mask = np.random.rand(*X_input.shape) < 0.1
    X_input[mask] = np.nan

    # 3. Initialize SAITS
    saits = SAITS(
        n_steps=WINDOW_SIZE,
        n_features=X.shape[2],
        n_layers=2,
        d_model=256,
        d_ffn=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=20,
        batch_size=256,
        patience=3,
        device="cuda",
        saving_path="./models/saits"
    )

    # 4. train
    print("開始訓練 SAITS...")
    saits.fit({"X": X_input})

    # --- 修改開始：手動分批填補 ---
    print("開始填補 (分批執行以避免 VRAM 爆炸)...")

    # 設定推論時的 Batch Size (可以比訓練大，例如 256 或 512，視顯卡記憶體而定)
    infer_batch_size = 512
    n_samples = len(X_input)
    imputation_list = []

    # 使用迴圈分批處理
    for i in range(0, n_samples, infer_batch_size):
        # 1. 切出這批資料
        batch_X = X_input[i: i + infer_batch_size]

        # 2. 讓模型填補這批資料
        # saits.impute 會回傳 numpy array，且通常已自動轉回 CPU
        batch_res = saits.impute({"X": batch_X})

        # 3. 存入列表
        imputation_list.append(batch_res)

    # 4. 將所有批次結果合併
    imputation = np.concatenate(imputation_list, axis=0)
    print(f"填補完成，總形狀: {imputation.shape}")
    # --- 修改結束 ---


    saits.save("./models/saits.pypots")
    # 5. Result
    sample_idx = 0
    restored_sample = scaler.inverse_transform(imputation[sample_idx])

    print(f"\nSAITS 填補完成。第一筆資料特徵預覽: {restored_sample[0][:5]}")


    # 為了繪圖，我們要把這三個都轉回真實數值 (Temperature, Pressure...)
    X_original_inv = scaler.inverse_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    X_imputed_inv = scaler.inverse_transform(imputation.reshape(-1, X.shape[2])).reshape(X.shape)

    # 對於含有 NaN 的 X_input，inverse_transform 可能會報錯或填補奇怪的值
    # 我們可以用一個小技巧：先填補 0 再轉換，然後把原本是 NaN 的地方再設回 NaN
    X_input_filled = np.nan_to_num(X_input, nan=0)
    X_input_inv = scaler.inverse_transform(X_input_filled.reshape(-1, X.shape[2])).reshape(X.shape)
    # 把 NaN 放回去 (根據原始 mask)
    mask_loc = np.isnan(X_input)
    X_input_inv[mask_loc] = np.nan

    # 2. 呼叫繪圖函數
    # 假設我們想看第 0 個樣本，第 1 個特徵 (通常第 0 是氣壓, 第 1 是溫度，看你的 CSV 順序)
    target_feature_idx = 1  # 試試看換成 0 或 2
    plot_imputation_result(
        X_original_inv,
        X_input_inv,
        X_imputed_inv,
        feature_names,
        sample_idx=0,
        feature_idx=target_feature_idx,
        model_name="saits"
    )

