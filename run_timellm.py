import numpy as np
import torch
import pypots
from pypots.imputation import TimeLLM
from data_loader import load_and_process_data
from plot import plot_imputation_result

if __name__ == '__main__':
    # 清理 VRAM
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
    # 注意：我們使用 GPT2 作為後端，這樣 12GB VRAM 才跑得動
    # 如果你堅持要用 LLaMA，需要改 llm_model_type="LLaMA" 並且要有 24GB+ VRAM (或極其複雜的 4-bit 修改)
    timellm = TimeLLM(
        n_steps=WINDOW_SIZE,
        n_features=X.shape[2],

        # --- TimeLLM 核心設定 ---
        llm_model_type="GPT2",  # 使用 GPT2 (輕量化)
        d_llm=768,  # GPT2-base 的維度是 768 (LLaMA 是 4096)
        d_model=32,  # TimeLLM 內部的 Embedding 維度
        d_ffn=128,
        n_layers=2,  # TimeLLM 的層數 (不是 LLM 的層數)
        n_heads=4,
        patch_size=16,  # Patching 大小
        patch_stride=8,  # Patching 步長
        dropout=0.1,
        domain_prompt_content="Weather data including temperature, pressure, humidity, wind speed, and other meteorological features.",
        # ----------------------

        epochs=20,  # LLM 比較難 train，可以先跑 10 epoch 看看
        batch_size=16,  # 記憶體敏感，建議設小一點 (8 ~ 16)
        patience=3,
        optimizer=pypots.optim.Adam(lr=1e-4),  # LLM 通常用 AdamW
        device="cuda",
        saving_path="./models/timellm_gpt2"
    )

    # 4. Train
    print("開始訓練 TimeLLM (Backbone: GPT2)...")
    timellm.fit({"X": X_input})

    # 5. Test
    # testModel(X, X_input, scaler, feature_names, timellm, "timellm", 16)
