import numpy as np
from pypots.imputation import BRITS  # 1. 修改這裡：改為匯入 BRITS
from data_loader import load_and_process_data
from plot import plot_imputation_result

if __name__ == '__main__':
    FILE_PATH = './dataset/weather.csv'
    WINDOW_SIZE = 144

    # 1. Load data
    X, scaler, feature_names = load_and_process_data(FILE_PATH, WINDOW_SIZE)

    # 2. Simulate missing data (randomly mask 10% for testing)
    X_input = X.copy()
    mask = np.random.rand(*X_input.shape) < 0.1
    X_input[mask] = np.nan

    # 3. Initialize BRITS
    # BRITS 是 RNN 架構，所以不需要 Attention 的參數 (如 n_heads, d_model 等)
    # 取而代之的是 rnn_hidden_size

    brits = BRITS(
        n_steps=WINDOW_SIZE,
        n_features=X.shape[2],
        rnn_hidden_size=64,  # 設定 RNN 隱藏層大小
        epochs=10,
        batch_size=32,
        patience=3,
        device="cuda",
        saving_path="./models/brits"
    )

    MODEL_PATH = "./models/brits/20251230_T162145/BRITS.pypots"
    print(f"正在載入模型: {MODEL_PATH} ...")
    brits.load(MODEL_PATH)

    # 4. Train
    print("開始訓練 BRITS...")
    brits.fit({"X": X_input})
    brits.save("./models/brits/brits.pypots")

    # PyPOTS 的 impute 通常回傳填補後的完整矩陣
    imputation = brits.impute({"X": X_input})

    # 5. Result
    sample_idx = 0
    restored_sample = scaler.inverse_transform(imputation[sample_idx])

    print(f"\nBRITS 填補完成。第一筆資料特徵預覽: {restored_sample[0][:5]}")

    # --- 以下繪圖邏輯與模型無關，保持不變 ---

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
    target_feature_idx = 1
    plot_imputation_result(
        X_original_inv,
        X_input_inv,
        X_imputed_inv,
        feature_names,
        sample_idx=0,
        feature_idx=target_feature_idx,
        model_name="brits"
    )