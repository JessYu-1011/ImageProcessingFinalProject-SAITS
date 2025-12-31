import numpy as np
from plot import plot_imputation_result



def testModel(X, X_input, scaler, feature_names, model, model_name, batch_size: int = 32):
    # --- 修改開始：手動分批填補 ---
    print("開始填補 (分批執行以避免 VRAM 爆炸)...")

    # 設定推論時的 Batch Size (可以比訓練大，例如 256 或 512，視顯卡記憶體而定)
    infer_batch_size = batch_size
    n_samples = len(X_input)
    imputation_list = []

    # 使用迴圈分批處理
    for i in range(0, n_samples, infer_batch_size):
        # 1. 切出這批資料
        batch_X = X_input[i: i + infer_batch_size]

        # 2. 讓模型填補這批資料
        # saits.impute 會回傳 numpy array，且通常已自動轉回 CPU
        batch_res = model.impute({"X": batch_X})

        # 3. 存入列表
        imputation_list.append(batch_res)

    # 4. 將所有批次結果合併
    imputation = np.concatenate(imputation_list, axis=0)
    print(f"填補完成，總形狀: {imputation.shape}")

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
        model_name=model_name
    )
