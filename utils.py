import numpy as np
from plot import plot_imputation_result



def testModel(X, X_input, scaler, feature_names, model, model_name, batch_size: int = 32):
    # --- Modification start: Manual batch imputation ---
    print("Starting imputation (batch execution to avoid VRAM overflow)...")

    # Set inference Batch Size (can be larger than training, e.g., 256 or 512, depending on GPU memory)
    infer_batch_size = batch_size
    n_samples = len(X_input)
    imputation_list = []

    # Use loop to process in batches
    for i in range(0, n_samples, infer_batch_size):
        # 1. Extract this batch of data
        batch_X = X_input[i: i + infer_batch_size]

        # 2. Let the model impute this batch of data
        # saits.impute will return a numpy array and is usually automatically converted back to CPU
        batch_res = model.impute({"X": batch_X})

        # 3. Add to list
        imputation_list.append(batch_res)

    # 4. Merge all batch results
    imputation = np.concatenate(imputation_list, axis=0)
    print(f"Imputation completed, total shape: {imputation.shape}")

    # 5. Result
    sample_idx = 0
    restored_sample = scaler.inverse_transform(imputation[sample_idx])

    print(f"\nBRITS imputation completed. First data feature preview: {restored_sample[0][:5]}")

    # --- The following plotting logic is model-independent, keep unchanged ---

    # To plot, we need to convert all three back to real values (Temperature, Pressure...)
    X_original_inv = scaler.inverse_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    X_imputed_inv = scaler.inverse_transform(imputation.reshape(-1, X.shape[2])).reshape(X.shape)

    # For X_input with NaN, inverse_transform may error or fill strange values
    # We can use a trick: first fill 0 then transform, then set the original NaN positions back to NaN
    X_input_filled = np.nan_to_num(X_input, nan=0)
    X_input_inv = scaler.inverse_transform(X_input_filled.reshape(-1, X.shape[2])).reshape(X.shape)

    # Put NaN back (based on original mask)
    mask_loc = np.isnan(X_input)
    X_input_inv[mask_loc] = np.nan

    # 2. Call the plotting function
    target_feature_idx = 1
    plot_imputation_result(
        X_original_inv,
        X_input_inv,
        X_imputed_inv,
        feature_names,
        sample_idx=0,
        feature_idx=target_feature_idx,
        model_name=model_name
    )
