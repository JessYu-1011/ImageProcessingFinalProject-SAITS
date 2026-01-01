import numpy as np
# 1. Modify here: change to import GPVAE
from pypots.imputation import GPVAE
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
    # GPVAE is a VAE architecture with completely different parameters from Transformer
    gpvae = GPVAE(
        n_steps=WINDOW_SIZE,
        n_features=X.shape[2],
        latent_size=32,  # Latent Space Dimension
        encoder_sizes=(64, 64),  # Encoder hidden layer structure
        decoder_sizes=(64, 64),  # Decoder hidden layer structure
        kernel="cauchy",  # Gaussian Process kernel function, options: "cauchy", "diffusion", "rbf", "matern"
        beta=1.0,  # KL Divergence weight (Beta-VAE)
        epochs=20,
        batch_size=256,
        patience=3,
        device="cuda",
        saving_path="./models/gpvae"
    )

    # 4. Train
    print("Starting GPVAE training...")
    gpvae.fit({"X": X_input})

    # 5. Test
    # testModel(X, X_input, scaler, feature_names, gpvae, "gpvae", 32)
