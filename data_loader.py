import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view  # 引入這個神器


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
    # 使用 sliding_window_view 創建視圖，不複製資料
    # data_scaled shape: (time_steps, features)
    # output shape: (n_samples, window_size, features)

    # 這裡 axis=0 表示我們要在時間維度上滑動
    X = sliding_window_view(data_scaled, window_size, axis=0)

    # sliding_window_view 的預設輸出形狀通常是把視窗維度放在最後
    # 如果原始是 (T, F)，結果會是 (T - W + 1, F, W)
    # 我們需要 (N, W, F)，所以可能需要 swapaxes

    # 讓我們檢查一下 sliding_window_view 的行為:
    # 對於 shape (T, F)，視窗大小 W 在 axis 0
    # 結果 shape 會是 (T - W + 1, F, W)

    # 我們需要轉置成 (Samples, Window, Features)
    # 所以要把最後一個維度 (W) 搬到中間
    X = np.moveaxis(X, -1, 1)

    # 如果有 step > 1 的需求，可以使用切片 (這也是 view，不佔記憶體)
    if step > 1:
        X = X[::step]

    # 為了避免後續 PyTorch 轉換報錯 (Negative strides)，這裡可能需要做一次 copy
    # 但這個 copy 是在訓練前最後一刻做，或者讓 DataLoader 處理
    # 為了保險起見，我們可以在這裡做 copy，但因為記憶體已經省下很多中間過程，通常沒問題。
    # 如果資料真的超級大，連這份 copy 都放不下，那就要寫 PyTorch Dataset 在 getitem 時才切。
    X = X.copy()

    print(f"資料載入完成: 形狀 {X.shape}")
    return X, scaler, feature_names