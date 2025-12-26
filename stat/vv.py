import numpy as np
import pandas as pd
from pathlib import Path
import re
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def fit(division='42'):
    # 按文件名排序
    def sort_key(filename):
        # 查找文件名中的数字部分（例如 "data123.p" -> 123）
        match = re.search(r'(\d+)', filename)
        return int(match.group()) if match else 0

    dataset = Path(__file__).stem
    data = {}
    for y in ('train', 'val', 'test'):
        # 获取 y_hr
        df = pd.read_csv(Path(dataset) / 'datafile' / 'split' / (y + '_' + division + '.csv'),
                         dtype={'subject_ID': str})
        data['y_sbp_' + y] = df['sysbp'].values
        data['y_dbp_' + y] = df['diasbp'].values

        # 获取 X
        unique_subjects = df['subject_ID'].unique()
        x = []
        datafiles = Path(dataset) / 'datafile' / 'ppg'
        for subject in unique_subjects:
            datafile_path = datafiles / subject
            files = [f for f in datafile_path.iterdir() if f.is_file() and f.suffix == ".p"]
            sorted_files = sorted(files, key=lambda f: sort_key(f.name))
            for file in sorted_files:
                x.append(joblib.load(file))

        assert len(x) == len(df), f"Number of X {len(x)} does not match number of Y {len(df)} in {y} set in {dataset}"
        data['X_' + y] = np.vstack(x)

    print("Training models with grid search...")

    # Prepare features and labels for training
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']

    y_sbp_train = data['y_sbp_train']
    y_dbp_train = data['y_dbp_train']

    y_sbp_val = data['y_sbp_val']
    y_dbp_val = data['y_dbp_val']

    y_sbp_test = data['y_sbp_test']
    y_dbp_test = data['y_dbp_test']

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid for Ridge regression
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'cholesky', 'sparse_cg']
    }

    # Combine training and validation sets for grid search
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_sbp_train_val = np.concatenate([y_sbp_train, y_sbp_val])
    y_dbp_train_val = np.concatenate([y_dbp_train, y_dbp_val])

    # Create validation indices for grid search (validation set indices in combined dataset)
    val_indices = list(range(len(X_train_scaled), len(X_train_val)))
    train_indices = list(range(len(X_train_scaled)))

    # Custom CV split for grid search (train on train set, validate on val set)
    custom_cv = [(train_indices, val_indices)]

    print("Optimizing SBP model...")
    # Grid search for SBP model
    ridge_sbp = Ridge(random_state=42)
    grid_search_sbp = GridSearchCV(
        ridge_sbp,
        param_grid,
        cv=custom_cv,
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search_sbp.fit(X_train_val, y_sbp_train_val)

    print("Optimizing DBP model...")
    # Grid search for DBP model
    ridge_dbp = Ridge(random_state=42)
    grid_search_dbp = GridSearchCV(
        ridge_dbp,
        param_grid,
        cv=custom_cv,
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search_dbp.fit(X_train_val, y_dbp_train_val)

    print(f"\nBest SBP parameters: {grid_search_sbp.best_params_}")
    print(f"Best SBP validation MAE: {-grid_search_sbp.best_score_:.3f}")

    print(f"\nBest DBP parameters: {grid_search_dbp.best_params_}")
    print(f"Best DBP validation MAE: {-grid_search_dbp.best_score_:.3f}")

    # Train final models on training set only with best parameters
    final_ridge_sbp = Ridge(**grid_search_sbp.best_params_, random_state=42)
    final_ridge_dbp = Ridge(**grid_search_dbp.best_params_, random_state=42)

    final_ridge_sbp.fit(X_train_scaled, y_sbp_train)
    final_ridge_dbp.fit(X_train_scaled, y_dbp_train)

    # Make predictions on validation set
    sbp_val_pred = final_ridge_sbp.predict(X_val_scaled)
    dbp_val_pred = final_ridge_dbp.predict(X_val_scaled)

    # Make predictions on test set
    sbp_test_pred = final_ridge_sbp.predict(X_test_scaled)
    dbp_test_pred = final_ridge_dbp.predict(X_test_scaled)


    # Calculate validation metrics
    sbp_val_mae = mean_absolute_error(y_sbp_val, sbp_val_pred)
    dbp_val_mae = mean_absolute_error(y_dbp_val, dbp_val_pred)

    # Calculate test metrics
    sbp_test_mae = mean_absolute_error(y_sbp_test, sbp_test_pred)
    dbp_test_mae = mean_absolute_error(y_dbp_test, dbp_test_pred)

    print("\n=== Validation Results (with optimized hyperparameters) ===")
    print(f"SBP - MAE: {sbp_val_mae:.3f}")
    print(f"DBP - MAE: {dbp_val_mae:.3f}")

    print("\n=== Test Results ===")
    print(f"SBP - MAE: {sbp_test_mae:.3f}")
    print(f"DBP - MAE: {dbp_test_mae:.3f}")
    # %%
    # Bootstrap confidence intervals for test set performance
    from scipy import stats

    def bootstrap_mae(y_true, y_pred, n_bootstrap=500, confidence_level=0.95):
        """
        Compute bootstrap confidence intervals for MAE.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CI)

        Returns:
        --------
        dict : Dictionary containing MAE statistics
        """
        np.random.seed(42)
        n_samples = len(y_true)
        bootstrap_maes = []

        for i in range(n_bootstrap):
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Calculate MAE for this bootstrap sample
            mae_boot = mean_absolute_error(y_true_boot, y_pred_boot)
            bootstrap_maes.append(mae_boot)

        bootstrap_maes = np.array(bootstrap_maes)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_maes, lower_percentile)
        ci_upper = np.percentile(bootstrap_maes, upper_percentile)
        mean_mae = np.mean(bootstrap_maes)
        std_mae = np.std(bootstrap_maes)

        return {
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_maes': bootstrap_maes
        }

    print("Computing 95% confidence intervals through bootstrapping (500 runs)...")
    print("=" * 60)

    # Bootstrap for SBP test results
    sbp_bootstrap = bootstrap_mae(y_sbp_test, sbp_test_pred, n_bootstrap=500)

    print("SBP Test Results:")
    print(f"  Original MAE: {sbp_test_mae:.3f}")
    print(f"  Bootstrap Mean MAE: {sbp_bootstrap['mean_mae']:.3f} ± {sbp_bootstrap['std_mae']:.3f}")
    print(f"  95% CI: [{sbp_bootstrap['ci_lower']:.3f}, {sbp_bootstrap['ci_upper']:.3f}]")

    # Bootstrap for DBP test results
    dbp_bootstrap = bootstrap_mae(y_dbp_test, dbp_test_pred, n_bootstrap=500)

    print("\nDBP Test Results:")
    print(f"  Original MAE: {dbp_test_mae:.3f}")
    print(f"  Bootstrap Mean MAE: {dbp_bootstrap['mean_mae']:.3f} ± {dbp_bootstrap['std_mae']:.3f}")
    print(f"  95% CI: [{dbp_bootstrap['ci_lower']:.3f}, {dbp_bootstrap['ci_upper']:.3f}]")