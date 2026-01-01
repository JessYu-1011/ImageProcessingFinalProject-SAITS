import numpy as np
import torch
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import load_and_process_data

# Import all potentially used models
from pypots.imputation import SAITS, CSDI, BRITS, TimesNet, GPVAE


# ==========================================
# 1. Model Factory
# ==========================================
def get_model(model_name, model_path, n_steps, n_features, device="cuda"):
    """
    Initialize the corresponding PyPOTS model and load weights based on model name and data shape.
    """
    model_name = model_name.lower()

    print(f"Initializing model architecture: {model_name.upper()} ...")

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
        raise ValueError(f"Unsupported model name: {model_name}")

    # 載入權重
    if os.path.exists(model_path):
        print(f"Loading weights: {model_path}")
        model.load(model_path)
        print("✅ Model loaded successfully!")
    else:
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    return model


# ==========================================
# 2. Helper Function: Batch Imputation
# ==========================================
def impute_in_chunks(model, data, chunk_size=256, model_name=""):
    num_samples = len(data)
    results = []
    print(f"Performing batch imputation on {num_samples} samples (Chunk size: {chunk_size})...")

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
            print(f"  > Processed {min(i + chunk_size, num_samples)} / {num_samples}")

    return np.concatenate(results, axis=0)


# ==========================================
# 3. 評估指標計算
# ==========================================
def calculate_metrics(original, imputed, mask):
    """Only calculate error for the masked portion"""
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
    TARGET_MODEL_NAME = "brits"
    MODEL_WEIGHT_PATH = "./models/brits/20251231_T111912/BRITS.pypots"

    DATA_PATH = "./dataset/weather.csv"
    WINDOW_SIZE = 144
    CHUNK_SIZE = 256
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.mps.is_available():
        DEVICE = "mps"

    # === Modification point 1: Set the missing rate list to test ===
    MISSING_RATES = [0.3, 0.5, 0.7]
    # =====================================

    # 設定結果儲存目錄
    SAVE_DIR = "./evaluation_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Unified timestamp, so all experiment files in this batch have the same time prefix
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== Preparing to start multi-rate evaluation process: {TARGET_MODEL_NAME.upper()} ===")
    print(f"Planned Missing Rates to test: {MISSING_RATES}")

    # 1. Load data (this part doesn't need to be repeated in the loop)
    X_intact, scaler, feature_names = load_and_process_data(DATA_PATH, WINDOW_SIZE)
    n_samples, n_steps, n_features = X_intact.shape

    # 2. Get model (load the model only once)
    try:
        model = get_model(
            TARGET_MODEL_NAME,
            MODEL_WEIGHT_PATH,
            n_steps=WINDOW_SIZE,
            n_features=X_intact.shape[2],
            device=DEVICE
        )
    except Exception as e:
        print(f"Model loading failed: {e}")
        exit()

    # === Modification point 2: Start loop to iterate through different missing rates ===
    for current_rate in MISSING_RATES:
        print("\n" + "#" * 60)
        print(f"🚀 Starting test Missing Rate: {current_rate * 100}%")
        print("#" * 60)

        # 3. Generate test data
        X_test_missing = X_intact.copy()

        # Set seed to ensure the mask for 0.3 is the same each time (reproducibility)
        np.random.seed(42)
        missing_mask = np.random.rand(*X_test_missing.shape) < current_rate
        X_test_missing[missing_mask] = np.nan
        print(f"Randomly masked {current_rate * 100}% of data (Seed=42)")

        # 4. Execute imputation
        current_chunk = CHUNK_SIZE
        imputation = impute_in_chunks(model, X_test_missing, chunk_size=current_chunk, model_name=TARGET_MODEL_NAME)

        # 5. Inverse normalization
        print("Performing inverse normalization...")
        flat_intact = X_intact.reshape(-1, n_features)
        flat_imputed = imputation.reshape(-1, n_features)
        real_intact = scaler.inverse_transform(flat_intact).reshape(n_samples, n_steps, n_features)
        real_imputed = scaler.inverse_transform(flat_imputed).reshape(n_samples, n_steps, n_features)

        # 6. Calculate error and save results
        report_data = []

        # Calculate overall error
        mae, rmse = calculate_metrics(real_intact, real_imputed, missing_mask)

        print("-" * 50)
        print(f"【Evaluation Report | Rate: {current_rate}】")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print("-" * 50)

        report_data.append({
            "Feature": "OVERALL (Average)",
            "MAE": mae,
            "RMSE": rmse,
            "Missing_Rate": current_rate  # Record which Rate this is in CSV
        })

        # Calculate error for each feature
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

        # === Modification point 3: Add rate to filename when saving ===
        # Save CSV report
        df_report = pd.DataFrame(report_data)
        # Filename example: report_csdi_rate0.3_20251231.csv
        csv_filename = f"{SAVE_DIR}/report_{TARGET_MODEL_NAME}_rate{current_rate}_{TIMESTAMP}.csv"
        df_report.to_csv(csv_filename, index=False)
        print(f"[Save] Evaluation report saved to: {csv_filename}")

        # Save imputed raw data (numpy array)
        # Filename example: imputed_csdi_rate0.3_20251231.npy
        npy_filename = f"{SAVE_DIR}/imputed_{TARGET_MODEL_NAME}_rate{current_rate}_{TIMESTAMP}.npy"
        np.save(npy_filename, imputation)
        print(f"[Save] Imputation result matrix saved to: {npy_filename}")

    print("\n✅ All missing rate tests completed!")