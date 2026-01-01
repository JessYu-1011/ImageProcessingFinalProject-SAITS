import numpy as np
from pypots.imputation import SAITS
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

    # 3. Initialize SAITS
    saits = SAITS(
        n_steps=WINDOW_SIZE,
        n_features=X.shape[2],
        n_layers=2,
        d_model=256,
        d_ffn=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=20,
        batch_size=256,
        patience=3,
        device="cuda",
        saving_path="./models/saits"
    )

    # 4. train
    print("Starting SAITS training...")
    saits.fit({"X": X_input})

    # 5. Test
    # testModel(X, X_input, scaler, feature_names, saits, "saits", 32)

