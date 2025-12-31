import numpy as np
from pypots.imputation import CSDI
from data_loader import load_and_process_data
from utils import testModel


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
        n_channels=32,
        n_heads=8,

        # --- 補上這三個缺少的嵌入維度參數 ---
        d_time_embedding=64,  # [必要] 時間嵌入的維度 (通常設 128 或 64)
        d_feature_embedding=64,  # [必要] 特徵嵌入的維度
        d_diffusion_embedding=64,  # [必要] 擴散步驟的嵌入維度
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

    # 5. Test
    # testModel(X, X_input, scaler, feature_names, csdi, "csdi", 32)
