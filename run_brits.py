import numpy as np
# 1. Modify here: change to import BRITS
from pypots.imputation import BRITS
from data_loader import load_and_process_data
from utils import testModel


if __name__ == '__main__':
    FILE_PATH = './dataset/weather.csv'
    WINDOW_SIZE = 144

    # 1. Load data
    X, scaler, feature_names = load_and_process_data(FILE_PATH, WINDOW_SIZE)

    # 2. Simulate missing data (randomly mask 10% for testing)
    X_input = X.copy()
    mask = np.random.rand(*X_input.shape) < 0.1
    X_input[mask] = np.nan

    # 3. Initialize BRITS
    # BRITS is an RNN architecture, so it doesn't need Attention parameters (like n_heads, d_model, etc.)
    # Instead, it uses rnn_hidden_size

    brits = BRITS(
        n_steps=WINDOW_SIZE,
        n_features=X.shape[2],
        rnn_hidden_size=64,  # Set RNN hidden layer size
        epochs=20,
        batch_size=256,
        patience=3,
        device="cuda",
        saving_path="./models/brits"
    )

    # 4. Train
    print("Starting BRITS training...")
    brits.fit({"X": X_input})


    # 5. Test
    # testModel(X, X_input, scaler, feature_names, brits, "brits", 32)
