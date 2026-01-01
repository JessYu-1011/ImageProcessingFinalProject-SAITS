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

    # 3. Initialize CSDI (corrected parameters)
    # 3. Initialize CSDI (RTX 4070 optimized version)
    csdi = CSDI(
        n_features=X.shape[2],
        n_steps=WINDOW_SIZE,

        # --- Model architecture optimization ---
        n_layers=3,  # Changed from 4 to 3 (sufficient)
        n_channels=32,  # Keep at 32
        n_heads=4,  # Changed from 8 to 4 (reduce matrix computation)

        d_time_embedding=64,
        d_feature_embedding=64,
        d_diffusion_embedding=64,

        # --- [Key 1] Reduce diffusion steps ---
        # 50 steps is for high-quality image generation, but for numerical imputation 20 steps usually work just as well
        n_diffusion_steps=20,

        target_strategy="random",
        is_unconditional=False,

        epochs=20,  # Since batch is larger, fewer updates, slightly increase epochs to compensate

        # --- [Key 2] Aggressively increase Batch Size ---
        # Your 4070 has 12GB, running a channels=32 model,
        # Batch Size of 128 is too conservative, increase to 1024!
        batch_size=128,

        patience=5,
        device="cuda",
        saving_path="./models/csdi"
    )
    # 4. train
    print("Starting CSDI training...")
    csdi.fit({"X": X_input})

    # 5. Test
    # testModel(X, X_input, scaler, feature_names, csdi, "csdi", 32)
