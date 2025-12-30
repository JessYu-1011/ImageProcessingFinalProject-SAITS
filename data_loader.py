import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_process_data(file_path, window_size=144, step=1):
    """
    Read CSV return processed (X, scaler, feature_names)
    """
    df = pd.read_csv(file_path)

    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    # Only number column
    data_values = df.select_dtypes(include=[np.number]).values
    feature_names = df.select_dtypes(include=[np.number]).columns

    # Normalization
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_values)

    # 3. Sliding Window
    # output shape: (samples, time_steps, features)
    n_samples = (len(data_scaled) - window_size) // step + 1

    # use numpy stride_tricks to split
    sub_windows = (
            np.expand_dims(np.arange(window_size), 0) +
            np.expand_dims(np.arange(n_samples * step, step=step), 0).T
    )
    X = data_scaled[sub_windows]

    print(f"資料載入完成: 形狀 {X.shape}")
    return X, scaler, feature_names