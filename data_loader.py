import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# Import this powerful stride tricks tool
from numpy.lib.stride_tricks import sliding_window_view


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

    # 3. Efficient Sliding Window (Memory Efficient!)
    # Use sliding_window_view to create a view without copying data
    # data_scaled shape: (time_steps, features)
    # output shape: (n_samples, window_size, features)

    # axis=0 means we slide along the time dimension
    X = sliding_window_view(data_scaled, window_size, axis=0)

    # sliding_window_view's default output shape usually places the window dimension at the end
    # If the original is (T, F), the result will be (T - W + 1, F, W)
    # We need (N, W, F), so we may need swapaxes

    # Let's check the behavior of sliding_window_view:
    # For shape (T, F), window size W on axis 0
    # The result shape will be (T - W + 1, F, W)

    # We need to transpose to (Samples, Window, Features)
    # So we move the last dimension (W) to the middle
    X = np.moveaxis(X, -1, 1)

    # If step > 1 is needed, we can use slicing (this is also a view, doesn't occupy memory)
    if step > 1:
        X = X[::step]

    # To avoid PyTorch conversion errors (Negative strides), we may need to copy here
    # But this copy is done at the last moment before training, or let DataLoader handle it
    # For safety, we can copy here, but since we've saved a lot of memory in intermediate steps, it's usually fine
    # If the data is really huge and even this copy doesn't fit, we need to write a PyTorch Dataset that slices in getitem
    X = X.copy()

    print(f"Data loaded successfully: shape {X.shape}")
    return X, scaler, feature_names