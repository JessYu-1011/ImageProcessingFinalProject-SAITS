import numpy as np
from pypots.imputation import CSDI
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

    # 3. Initialize CSDI (修正參數版)
    csdi = CSDI(
        n_features=X.shape[2],
        n_steps=WINDOW_SIZE,  # [必要] 必須指定時間步長 (這裡就是 144)
        n_layers=4,
        n_channels=64,
        n_heads=8,

        # --- 補上這三個缺少的嵌入維度參數 ---
        d_time_embedding=128,  # [必要] 時間嵌入的維度 (通常設 128 或 64)
        d_feature_embedding=128,  # [必要] 特徵嵌入的維度
        d_diffusion_embedding=128,  # [必要] 擴散步驟的嵌入維度
        # ----------------------------------

        n_diffusion_steps=50,
        target_strategy="random",  # CSDI 訓練時隨機遮罩的策略 (建議加上)
        is_unconditional=False,
        epochs=20,
        batch_size=128,
        patience=3,
        device="cuda",
        saving_path="./models/csdi"
    )

    # 4. train
    print("開始訓練 CSDI ...")
    csdi.fit({"X": X_input})

    # --- 修改開始：手動分批填補 ---
    print("開始填補 (分批執行以避免 VRAM 爆炸)...")
    print("注意：Diffusion 模型推論速度比 SAITS 慢很多，這是正常的。")

    # 設定推論時的 Batch Size
    # CSDI 運算量比 SAITS 大，建議 infer_batch_size 設小一點 (例如 128 或 256)
    infer_batch_size = 256
    n_samples = len(X_input)
    imputation_list = []

    # 使用迴圈分批處理
    for i in range(0, n_samples, infer_batch_size):
        # 1. 切出這批資料
        batch_X = X_input[i: i + infer_batch_size]

        # 2. 讓模型填補這批資料
        # n_sampling_times=1 代表對每個缺失點只採樣一次 (通常用於確定性填補)
        batch_res = csdi.impute({"X": batch_X}, n_sampling_times=1)

        # 3. 存入列表
        imputation_list.append(batch_res)

        # 顯示進度 (因為 Diffusion 跑很慢，建議印出來看)
        if (i // infer_batch_size) % 5 == 0:
            print(f"  已處理 {min(i + infer_batch_size, n_samples)} / {n_samples} 筆...")

    # 4. 將所有批次結果合併
    imputation = np.concatenate(imputation_list, axis=0)
    print(f"填補完成，總形狀: {imputation.shape}")
    # --- 修改結束 ---

    csdi.save("./models/csdi.pypots")

    # 5. Result
    sample_idx = 0
    # CSDI 輸出有時會多一個維度 (samples)，確保維度正確再轉換
    if imputation.ndim == 4:
        imputation = imputation.mean(axis=1)  # 如果有多重採樣，取平均

    restored_sample = scaler.inverse_transform(imputation[sample_idx])

    print(f"\nCSDI 填補完成。第一筆資料特徵預覽: {restored_sample[0][:5]}")

    # 為了繪圖，我們要把這三個都轉回真實數值 (Temperature, Pressure...)
    X_original_inv = scaler.inverse_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    X_imputed_inv = scaler.inverse_transform(imputation.reshape(-1, X.shape[2])).reshape(X.shape)

    # 對於含有 NaN 的 X_input，inverse_transform 可能會報錯或填補奇怪的值
    X_input_filled = np.nan_to_num(X_input, nan=0)
    X_input_inv = scaler.inverse_transform(X_input_filled.reshape(-1, X.shape[2])).reshape(X.shape)
    # 把 NaN 放回去
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
        model_name="CSDI"
    )