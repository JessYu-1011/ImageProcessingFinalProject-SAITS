import numpy as np
import torch
import pypots
from pypots.imputation import TimeLLM
from data_loader import load_and_process_data
from plot import plot_imputation_result

if __name__ == '__main__':
    # Clear VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    FILE_PATH = './dataset/weather.csv'
    WINDOW_SIZE = 144

    # 1. Load data
    X, scaler, feature_names = load_and_process_data(FILE_PATH, WINDOW_SIZE)

    # 2. Simulate missing data
    X_input = X.copy()
    mask = np.random.rand(*X_input.shape) < 0.1
    X_input[mask] = np.nan

    # 3. Initialize TimeLLM
    # Note: We use GPT2 as the backend, so 12GB VRAM can handle it
    # If you insist on using LLaMA, you need to change llm_model_type="LLaMA" and require 24GB+ VRAM (or extremely complex 4-bit modifications)
    timellm = TimeLLM(
        n_steps=WINDOW_SIZE,
        n_features=X.shape[2],

        # --- TimeLLM Core Settings ---
        llm_model_type="GPT2",  # Use GPT2 (lightweight)
        d_llm=768,  # GPT2-base dimension is 768 (LLaMA is 4096)
        d_model=32,  # TimeLLM internal embedding dimension
        d_ffn=128,
        n_layers=2,  # TimeLLM layers (not LLM layers)
        n_heads=4,
        patch_size=16,  # Patching size
        patch_stride=8,  # Patching stride
        dropout=0.1,
        domain_prompt_content="Weather data including temperature, pressure, humidity, wind speed, and other meteorological features.",
        # ----------------------

        epochs=20,  # LLM is harder to train, can start with 10 epochs to see
        batch_size=16,  # Memory sensitive, recommend setting smaller (8 ~ 16)
        patience=3,
        optimizer=pypots.optim.Adam(lr=1e-4),  # LLM usually uses AdamW
        device="cuda",
        saving_path="./models/timellm_gpt2"
    )

    # 4. Train
    print("Starting TimeLLM training (Backbone: GPT2)...")
    timellm.fit({"X": X_input})

    # 5. Test
    # testModel(X, X_input, scaler, feature_names, timellm, "timellm", 16)
