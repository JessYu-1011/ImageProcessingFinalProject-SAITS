import numpy as np
from pypots.imputation import GPVAE  # 1. 修改這裡：改為匯入 GPVAE
from data_loader import load_and_process_data
from utils import testModel


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

    # 5. Test
    # testModel(X, X_input, scaler, feature_names, gpvae, "gpvae", 32)
