import matplotlib.pyplot as plt

def plot_imputation_result(original, missing, imputed, feature_names, sample_idx=0, feature_idx=1, model_name=""):
    """
    original: 完整的原始資料 (反標準化後)
    missing:  帶有 NaN 的輸入資料 (反標準化後)
    imputed:  模型填補後的結果 (反標準化後)
    feature_names: 特徵名稱列表
    sample_idx: 要畫第幾個樣本 (Window)
    feature_idx: 要畫第幾個特徵 (例如 1 代表 溫度)
    """

    # 取出特定樣本與特徵的數據
    # 形狀都是 (TimeSteps,)
    org_data = original[sample_idx, :, feature_idx]
    imp_data = imputed[sample_idx, :, feature_idx]

    # 建立一個畫布
    plt.figure(figsize=(12, 6))

    # 1. 畫出模型填補的結果 (紅色實線)
    # 包含了「原本有的」和「填補出來的」
    plt.plot(imp_data, color='red', label='Imputed (Model Output)', linestyle='-', alpha=0.7)

    # 2. 畫出原始真實數據 (綠色虛線)
    # 這讓我們知道「正確答案」是什麼
    plt.plot(org_data, color='green', label='Ground Truth (Original)', linestyle='--', alpha=0.6)

    # 3. 畫出觀測到的數據 (藍色點/線)
    # 這些是模型「看得到」的數據。斷掉的地方就是 NaN。
    # 注意：反標準化時 NaN 可能會被填成數字，所以這裡我們重新用 mask 過濾一下確保不畫出來
    # 但為了簡單，如果你傳入的 missing 裡面有 NaN，matplotlib 會自動斷開不畫，這正是我們要的
    plt.plot(missing[sample_idx, :, feature_idx], color='blue', label='Observed (Input)', linewidth=2)

    plt.title(f"Imputation Result: Sample {sample_idx}, Feature '{feature_names[feature_idx]}'")
    plt.xlabel("Time Steps (10 mins per step)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./figures/{model_name}.png")
    plt.show()


