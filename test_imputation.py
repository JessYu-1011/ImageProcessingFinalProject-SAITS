import numpy as np
import torch
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import load_and_process_data

# 引入所有可能用到的模型
from pypots.imputation import SAITS, CSDI, BRITS, TimesNet, GPVAE


# ==========================================
# 1. 模型初始化工廠 (Model Factory)
# ==========================================
def get_model(model_name, model_path, n_steps, n_features, device="cuda"):
    """
    根據模型名稱與資料形狀，初始化對應的 PyPOTS 模型並載入權重。
    """
    model_name = model_name.lower()

    print(f"正在初始化模型架構: {model_name.upper()} ...")

    if model_name == 'saits':
        model = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=2,
            d_model=256,
            d_ffn=128,
            n_heads=4,
            d_k=64,
            d_v=64,
            dropout=0.1,
            device=device
        )
    elif model_name == 'brits':
        model = BRITS(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=64,
            device=device
        )
    elif model_name == 'csdi':
        model = CSDI(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=4,
            n_channels=32,
            n_heads=8,
            d_time_embedding=64,
            d_feature_embedding=64,
            d_diffusion_embedding=64,
            n_diffusion_steps=50,
            device=device
        )
    elif model_name == 'timesnet':
        model = TimesNet(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=2,
            top_k=5,
            d_model=64,
            d_ffn=128,
            n_heads=4,
            device=device
        )
    elif model_name == 'gpvae':
        model = GPVAE(
            n_steps=n_steps,
            n_features=n_features,
            latent_size=32,
            encoder_sizes=(64, 64),
            decoder_sizes=(64, 64),
            kernel="cauchy",
            device=device
        )
    else:
        raise ValueError(f"不支援的模型名稱: {model_name}")

    # 載入權重
    if os.path.exists(model_path):
        print(f"正在載入權重: {model_path}")
        model.load(model_path)
        print("✅ 模型載入成功！")
    else:
        raise FileNotFoundError(f"❌ 找不到模型檔案: {model_path}")

    return model


# ==========================================
# 2. 輔助函式：分批填補
# ==========================================
def impute_in_chunks(model, data, chunk_size=256, model_name=""):
    num_samples = len(data)
    results = []
    print(f"正在分批填補 {num_samples} 筆資料 (Chunk size: {chunk_size})...")

    for i in range(0, num_samples, chunk_size):
        chunk = data[i: i + chunk_size]

        if "csdi" in model_name.lower():
            chunk_res = model.impute({"X": chunk}, n_sampling_times=1)
            if chunk_res.ndim == 4:
                chunk_res = chunk_res.mean(axis=1)
        else:
            chunk_res = model.impute({"X": chunk})

        results.append(chunk_res)

        if (i // chunk_size) % 10 == 0:
            print(f"  > 已處理 {min(i + chunk_size, num_samples)} / {num_samples}")

    return np.concatenate(results, axis=0)


# ==========================================
# 3. 評估指標計算
# ==========================================
def calculate_metrics(original, imputed, mask):
    """只計算 mask 部分的誤差"""
    org_values = original[mask]
    imp_values = imputed[mask]

    mae = mean_absolute_error(org_values, imp_values)
    rmse = np.sqrt(mean_squared_error(org_values, imp_values))
    return mae, rmse


# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    # 設定參數
    TARGET_MODEL_NAME = "csdi"
    MODEL_WEIGHT_PATH = "./models/csdi/20251231_T012023/CSDI.pypots"

    DATA_PATH = "./dataset/weather.csv"
    WINDOW_SIZE = 144
    CHUNK_SIZE = 1024
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # === 修改點 1: 設定要測試的缺失率列表 ===
    MISSING_RATES = [0.3, 0.5, 0.7]
    # =====================================

    # 設定結果儲存目錄
    SAVE_DIR = "./evaluation_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 統一一個時間戳記，讓這批實驗的檔案有一樣的前綴時間
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== 準備開始多重缺失率評估流程: {TARGET_MODEL_NAME.upper()} ===")
    print(f"計畫測試的 Missing Rates: {MISSING_RATES}")

    # 1. 載入資料 (這部分不用在迴圈內重複做)
    X_intact, scaler, feature_names = load_and_process_data(DATA_PATH, WINDOW_SIZE)
    n_samples, n_steps, n_features = X_intact.shape

    # 2. 獲取模型 (模型載入一次就好)
    try:
        model = get_model(
            TARGET_MODEL_NAME,
            MODEL_WEIGHT_PATH,
            n_steps=WINDOW_SIZE,
            n_features=X_intact.shape[2],
            device=DEVICE
        )
    except Exception as e:
        print(f"模型載入失敗: {e}")
        exit()

    # === 修改點 2: 開始迴圈遍歷不同的缺失率 ===
    for current_rate in MISSING_RATES:
        print("\n" + "#" * 60)
        print(f"🚀 開始測試 Missing Rate: {current_rate * 100}%")
        print("#" * 60)

        # 3. 產生測試資料
        X_test_missing = X_intact.copy()

        # 設定 seed 確保這一次 0.3 的遮罩跟下次跑 0.3 的遮罩是一樣的 (可重現性)
        np.random.seed(42)
        missing_mask = np.random.rand(*X_test_missing.shape) < current_rate
        X_test_missing[missing_mask] = np.nan
        print(f"已隨機挖空 {current_rate * 100}% 的資料 (Seed=42)")

        # 4. 執行填補
        current_chunk = CHUNK_SIZE
        imputation = impute_in_chunks(model, X_test_missing, chunk_size=current_chunk, model_name=TARGET_MODEL_NAME)

        # 5. 反標準化
        print("正在反標準化...")
        flat_intact = X_intact.reshape(-1, n_features)
        flat_imputed = imputation.reshape(-1, n_features)
        real_intact = scaler.inverse_transform(flat_intact).reshape(n_samples, n_steps, n_features)
        real_imputed = scaler.inverse_transform(flat_imputed).reshape(n_samples, n_steps, n_features)

        # 6. 計算誤差與儲存結果
        report_data = []

        # 計算整體誤差
        mae, rmse = calculate_metrics(real_intact, real_imputed, missing_mask)

        print("-" * 50)
        print(f"【評估報告 | Rate: {current_rate}】")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print("-" * 50)

        report_data.append({
            "Feature": "OVERALL (Average)",
            "MAE": mae,
            "RMSE": rmse,
            "Missing_Rate": current_rate  # 在 CSV 中記錄這是哪個 Rate
        })

        # 計算各特徵誤差
        for i, name in enumerate(feature_names):
            feat_mask = missing_mask[:, :, i]
            if np.sum(feat_mask) == 0:
                feat_mae, feat_rmse = 0.0, 0.0
            else:
                feat_mae, feat_rmse = calculate_metrics(
                    real_intact[:, :, i],
                    real_imputed[:, :, i],
                    feat_mask
                )

            report_data.append({
                "Feature": name,
                "MAE": feat_mae,
                "RMSE": feat_rmse,
                "Missing_Rate": current_rate
            })

        # === 修改點 3: 存檔時檔名加上 rate 區分 ===
        # 儲存 CSV 報告
        df_report = pd.DataFrame(report_data)
        # 檔名範例: report_csdi_rate0.3_20251231.csv
        csv_filename = f"{SAVE_DIR}/report_{TARGET_MODEL_NAME}_rate{current_rate}_{TIMESTAMP}.csv"
        df_report.to_csv(csv_filename, index=False)
        print(f"[Save] 評估報告已儲存至: {csv_filename}")

        # 儲存填補後的 raw data (numpy array)
        # 檔名範例: imputed_csdi_rate0.3_20251231.npy
        npy_filename = f"{SAVE_DIR}/imputed_{TARGET_MODEL_NAME}_rate{current_rate}_{TIMESTAMP}.npy"
        np.save(npy_filename, imputation)
        print(f"[Save] 填補結果矩陣已儲存至: {npy_filename}")

    print("\n✅ 所有缺失率測試完成！")