import numpy as np
import torch
from pypots.imputation import SAITS
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import load_and_process_data

# ==========================================
# 1. 設定與參數
# ==========================================
MODEL_PATH = "my_best_model/saits_weather.pypots"  # 你存模型的路徑
DATA_PATH = "./dataset/weather.csv"  # 資料路徑
WINDOW_SIZE = 144
MISSING_RATE = 0.2  # 測試遮罩率：挖掉 20% 來考驗模型


# ==========================================
# 2. 輔助函式：分批填補 (避免 OOM)
# ==========================================
def impute_in_chunks(model, data_dict, chunk_size=500):
    full_X = data_dict["X"]
    num_samples = len(full_X)
    results = []
    print(f"正在分批填補 {num_samples} 筆資料 (Chunk size: {chunk_size})...")

    for i in range(0, num_samples, chunk_size):
        chunk = full_X[i: i + chunk_size]
        # 這裡會自動用 GPU 跑，但因為切小了所以不會爆顯存
        chunk_res = model.impute({"X": chunk})
        results.append(chunk_res)

    return np.concatenate(results, axis=0)


# ==========================================
# 3. 評估函式
# ==========================================
def calculate_metrics(original, imputed, mask):
    """
    只計算 mask 部份的誤差 (Ground Truth vs Imputed)
    original, imputed: 必須是已經 inverse_transform 過的真實數值
    mask: boolean array, True 代表該位置原本被挖空 (是我們要評估的對象)
    """
    # 只取出被挖空部分的數值
    org_values = original[mask]
    imp_values = imputed[mask]

    # 計算 MAE (平均絕對誤差)
    mae = mean_absolute_error(org_values, imp_values)

    # 計算 RMSE (均方根誤差)
    rmse = np.sqrt(mean_squared_error(org_values, imp_values))

    return mae, rmse


# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    # 1. 載入資料 (這就是 Ground Truth)
    X_intact, scaler, feature_names = load_and_process_data(DATA_PATH, WINDOW_SIZE)
    print(f"資料載入完成，原始形狀: {X_intact.shape}")

    # 2. 載入模型
    # 必須初始化一個結構一樣的模型才能 load 權重
    print("正在載入模型...")
    saits = SAITS(
        n_steps=WINDOW_SIZE,
        n_features=X_intact.shape[2],
        n_layers=2,
        d_model=256,
        d_ffn=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        device="cuda"  # 使用 GPU 加速
    )
    try:
        saits.load(MODEL_PATH)
        print("模型權重載入成功！")
    except FileNotFoundError:
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}，請確認路徑或先執行訓練。")
        exit()

    # 3. 產生測試資料 (製造人工缺失)
    X_test_missing = X_intact.copy()
    # 產生隨機遮罩 (True 代表要變 NaN)
    missing_mask = np.random.rand(*X_test_missing.shape) < MISSING_RATE
    X_test_missing[missing_mask] = np.nan

    print(f"已隨機挖空 {MISSING_RATE * 100}% 的資料作為測試集...")

    # 4. 執行填補
    imputation = impute_in_chunks(saits, {"X": X_test_missing}, chunk_size=256)

    # ==========================================
    # 5. 反標準化 (還原成真實物理量)
    # ==========================================
    print("正在進行反標準化與誤差計算...")

    # 因為 inverse_transform 吃 2D array，我們需要先 reshape
    n_samples, n_steps, n_features = X_intact.shape

    # 轉換 Ground Truth
    flat_intact = X_intact.reshape(-1, n_features)
    real_intact = scaler.inverse_transform(flat_intact).reshape(n_samples, n_steps, n_features)

    # 轉換 Imputed Result
    flat_imputed = imputation.reshape(-1, n_features)
    real_imputed = scaler.inverse_transform(flat_imputed).reshape(n_samples, n_steps, n_features)

    # ==========================================
    # 6. 計算並列印整體誤差
    # ==========================================
    mae, rmse = calculate_metrics(real_intact, real_imputed, missing_mask)

    print("-" * 50)
    print(f"【整體測試結果 (Missing Rate {MISSING_RATE * 100}%)】")
    print(f"MAE  (平均絕對誤差): {mae:.4f}")
    print(f"RMSE (均方根誤差): {rmse:.4f}")
    print("-" * 50)

    # ==========================================
    # 7. (進階) 查看每一個特徵個別的誤差
    # ==========================================
    print("【各特徵誤差詳情】")
    # 為了排版好看
    print(f"{'Feature Name':<20} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 46)

    for i, name in enumerate(feature_names):
        # 針對第 i 個特徵建立 mask
        feat_mask = missing_mask[:, :, i]

        # 如果該特徵剛好沒被挖到任何洞 (機率很低但防呆)，就跳過
        if np.sum(feat_mask) == 0:
            continue

        feat_mae, feat_rmse = calculate_metrics(
            real_intact[:, :, i],
            real_imputed[:, :, i],
            feat_mask
        )
        print(f"{name:<20} | {feat_mae:.4f}     | {feat_rmse:.4f}")