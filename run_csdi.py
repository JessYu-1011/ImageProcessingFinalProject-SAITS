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
    # 3. Initialize CSDI (RTX 4070 加速版)
    csdi = CSDI(
        n_features=X.shape[2],
        n_steps=WINDOW_SIZE,

        # --- 模型架構輕量化 ---
        n_layers=3,  # 從 4 改為 3 (夠用了)
        n_channels=32,  # 維持 32
        n_heads=4,  # 從 8 改為 4 (減少矩陣運算量)

        d_time_embedding=64,
        d_feature_embedding=64,
        d_diffusion_embedding=64,

        # --- [關鍵 1] 減少擴散步數 ---
        # 50 步是給高品質圖片生成的，數值填補 20 步通常效果一樣好
        n_diffusion_steps=20,

        target_strategy="random",
        is_unconditional=False,

        epochs=20,  # 因為 Batch 大了，更新次數變少，稍微增加 epoch 補償

        # --- [關鍵 2] 暴力加大 Batch Size ---
        # 你的 4070 有 12GB，跑 channels=32 的模型，
        # Batch Size 開 128 太浪費了，直接開到 1024！
        batch_size=128,

        patience=5,
        device="cuda",
        saving_path="./models/csdi"
    )
    # 4. train
    print("開始訓練 CSDI ...")
    csdi.fit({"X": X_input})

    # 5. Test
    # testModel(X, X_input, scaler, feature_names, csdi, "csdi", 32)
