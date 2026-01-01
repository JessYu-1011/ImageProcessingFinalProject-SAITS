import os.path

import matplotlib.pyplot as plt

def plot_imputation_result(original, missing, imputed, feature_names, sample_idx=0, feature_idx=1, model_name=""):
    """
    original: Complete original data (inverse normalized)
    missing:  Input data with NaN (inverse normalized)
    imputed:  Model imputation results (inverse normalized)
    feature_names: List of feature names
    sample_idx: Which sample (window) to plot
    feature_idx: Which feature to plot (e.g., 1 represents Temperature)
    """

    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    # Extract data for specific sample and feature
    # All shapes are (TimeSteps,)
    org_data = original[sample_idx, :, feature_idx]
    imp_data = imputed[sample_idx, :, feature_idx]

    # Create a canvas
    plt.figure(figsize=(12, 6))

    # 1. Plot the model imputation results (red solid line)
    # Includes "original" and "imputed" values
    plt.plot(imp_data, color='red', label='Imputed (Model Output)', linestyle='-', alpha=0.7)

    # 2. Plot the original ground truth data (green dashed line)
    # This shows us what the "correct answer" is
    plt.plot(org_data, color='green', label='Ground Truth (Original)', linestyle='--', alpha=0.6)

    # 3. Plot the observed data (blue points/line)
    # This is the data the model "can see". Gaps are where NaN values exist.
    # Note: When inverse normalizing, NaN might be filled with numbers, so we re-filter with mask to ensure they're not plotted
    # But for simplicity, if your missing has NaN, matplotlib will automatically break the line, which is exactly what we want
    plt.plot(missing[sample_idx, :, feature_idx], color='blue', label='Observed (Input)', linewidth=2)

    plt.title(f"Imputation Result: Sample {sample_idx}, Feature '{feature_names[feature_idx]}'")
    plt.xlabel("Time Steps (10 mins per step)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./figures/{model_name}.png")
    plt.show()


