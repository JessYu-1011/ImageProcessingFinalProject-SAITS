import numpy as np
from pypots.imputation import GPVAE  # 1. 修改這裡：改為匯入 GPVAE
from data_loader import load_and_process_data
from plot import plot_imputation_result

if __name__ == '__main__':
    FILE_PATH = './dataset/weather.csv'
    WINDOW_SIZE = 144

    # 1. Load data
    X, scaler, feature_names = load_and_process_data(FILE_PATH, WINDOW_SIZE)

    # 2. Simulate missing data
    X_input = X.copy()
    mask = np.random.rand(*X_input.shape) < 0.1
    X_input[mask] = np.nan

    # 3. Initialize GPVAE
    # GPVAE 是 VAE 架構，參數與 Transformer 完全不同
    gpvae = GPVAE(
        n_steps=WINDOW_SIZE,
        n_features=X.shape[2],
        latent_size=32,  # 隱變量空間的維度 (Latent Space Dimension)
        encoder_sizes=(64, 64),  # Encoder 的隱藏層結構
        decoder_sizes=(64, 64),  # Decoder 的隱藏層結構
        kernel="cauchy",  # 高斯過程的核函數，可選 "cauchy", "diffusion", "rbf", "matern"
        beta=1.0,  # KL Divergence 的權重 (Beta-VAE)
        epochs=20,
        batch_size=256,
        patience=3,
        device="cuda",
        saving_path="./models/gpvae"
    )

    # 4. Train
    print("開始訓練 GPVAE...")
    gpvae.fit({"X": X_input})

    # --- 手動分批填補 (GPVAE 計算量也很大，保留這個邏輯很好) ---
    print("開始填補 (分批執行以避免 VRAM 爆炸)...")

    infer_batch_size = 512
    n_samples = len(X_input)
    imputation_list = []

    for i in range(0, n_samples, infer_batch_size):
        batch_X = X_input[i: i + infer_batch_size]

        # GPVAE 的 impute 一樣接收 dict
        batch_res = gpvae.impute({"X": batch_X})
        imputation_list.append(batch_res)

    imputation = np.concatenate(imputation_list, axis=0)
    print(f"填補完成，總形狀: {imputation.shape}")
    # --- 修改結束 ---

    # 儲存模型 (注意副檔名習慣)
    # 雖然 PyPOTS 會在 saving_path 自動存，但手動存一個也不錯
    # gpvae.save("./models/gpvae_manual.pypots")

    # 5. Result
    sample_idx = 0
    restored_sample = scaler.inverse_transform(imputation[sample_idx])

    print(f"\nGPVAE 填補完成。第一筆資料特徵預覽: {restored_sample[0][:5]}")

    # --- 以下繪圖邏輯保持不變 ---
    X_original_inv = scaler.inverse_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    X_imputed_inv = scaler.inverse_transform(imputation.reshape(-1, X.shape[2])).reshape(X.shape)

    X_input_filled = np.nan_to_num(X_input, nan=0)
    X_input_inv = scaler.inverse_transform(X_input_filled.reshape(-1, X.shape[2])).reshape(X.shape)
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
        model_name="gpvae"  # 修改圖表標題
    )