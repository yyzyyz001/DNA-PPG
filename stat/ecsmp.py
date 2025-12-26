import numpy as np
import pandas as pd
from pathlib import Path
import re
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def fit():
    # 按文件名排序
    def sort_key(filename):
        # 查找文件名中的数字部分（例如 "data123.p" -> 123）
        match = re.search(r'(\d+)', filename)
        return int(match.group()) if match else 0

    dataset = Path(__file__).stem
    data = {}
    for y in ('train', 'val', 'test'):
        # 获取 y_hr
        df = pd.read_csv(Path(dataset) / 'datafile' / 'split' / (y + '_42.csv'), dtype={'subject_ID': str})
        data['y_tmd_' + y] = df['TMD'].values

        # 获取 X
        unique_subjects = df['subject_ID'].unique()
        print(unique_subjects)
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

    y_tmd_train = data['y_tmd_train']
    y_tmd_val = data['y_tmd_val']
    y_tmd_test = data['y_tmd_test']

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter grid for Logistic regression
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }

    # Combine training and validation sets for grid search
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_tmd_train_val = np.concatenate([y_tmd_train, y_tmd_val])

    # Create validation indices for grid search (validation set indices in combined dataset)
    val_indices = list(range(len(X_train_scaled), len(X_train_val)))
    train_indices = list(range(len(X_train_scaled)))

    # Custom CV split for grid search (train on train set, validate on val set)
    custom_cv = [(train_indices, val_indices)]

    print("Optimizing TMD_binary model...")
    # Grid search for TMD_binary model
    logistic_tmd = LogisticRegression(random_state=42)
    grid_search_tmd = GridSearchCV(
        logistic_tmd,
        param_grid,
        cv=custom_cv,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    grid_search_tmd.fit(X_train_val, y_tmd_train_val)

    print(f"\nBest TMD_binary parameters: {grid_search_tmd.best_params_}")
    print(f"Best TMD_binary validation AUROC: {grid_search_tmd.best_score_:.3f}")

    # Train final models on training set only with best parameters
    final_logistic_tmd = LogisticRegression(**grid_search_tmd.best_params_, random_state=42)

    final_logistic_tmd.fit(X_train_scaled, y_tmd_train)

    # Make predictions on validation set
    tmd_val_pred = final_logistic_tmd.predict_proba(X_val_scaled)[:, 1]

    # Make predictions on test set
    tmd_test_pred = final_logistic_tmd.predict_proba(X_test_scaled)[:, 1]

    # Calculate validation metrics
    tmd_val_auroc = roc_auc_score(y_tmd_val, tmd_val_pred)

    # Calculate test metrics
    tmd_test_auroc = roc_auc_score(y_tmd_test, tmd_test_pred)

    print("\n=== Validation Results (with optimized hyperparameters) ===")
    print(f"TMD_binary - AUROC: {tmd_val_auroc:.3f}")

    print("\n=== Test Results ===")
    print(f"TMD_binary - AUROC: {tmd_test_auroc:.3f}")

    def bootstrap_auroc(y_true, y_pred_proba, n_bootstrap=500, confidence_level=0.95):
        """
        Compute bootstrap confidence intervals for AUROC.

        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_pred_proba : array-like
            Predicted probabilities for positive class
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CI)

        Returns:
        --------
        dict : Dictionary containing AUROC statistics
        """
        np.random.seed(42)
        n_samples = len(y_true)
        bootstrap_aurocs = []

        for i in range(n_bootstrap):
            # Bootstrap sample with replacement
            max_attempts = 100  # Prevent infinite loop
            for attempt in range(max_attempts):
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                y_true_boot = y_true[indices]

                # Check if both classes are present
                unique_classes = np.unique(y_true_boot)
                if len(unique_classes) == 2:  # Both 0 and 1 present
                    break
            else:
                # If we couldn't get both classes after max_attempts, skip this bootstrap
                print(f"Warning: Skipping bootstrap sample {i} - couldn't obtain both classes")
                continue

            y_pred_boot = y_pred_proba[indices]

            # Calculate AUROC for this bootstrap sample
            auroc_boot = roc_auc_score(y_true_boot, y_pred_boot)
            bootstrap_aurocs.append(auroc_boot)

        if len(bootstrap_aurocs) == 0:
            print("Error: No valid bootstrap samples could be generated")
            return None

        bootstrap_aurocs = np.array(bootstrap_aurocs)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_aurocs, lower_percentile)
        ci_upper = np.percentile(bootstrap_aurocs, upper_percentile)
        mean_auroc = np.mean(bootstrap_aurocs)
        std_auroc = np.std(bootstrap_aurocs)

        return {
            'mean_auroc': mean_auroc,
            'std_auroc': std_auroc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_aurocs': bootstrap_aurocs,
            'n_valid_samples': len(bootstrap_aurocs)
        }

    print("Computing 95% confidence intervals through bootstrapping (500 runs)...")
    print("=" * 60)

    # Bootstrap for TMD test results
    tmd_bootstrap = bootstrap_auroc(y_tmd_test, tmd_test_pred, n_bootstrap=500)

    if tmd_bootstrap is not None:
        print("TMD_binary Test Results:")
        print(f"  Original AUROC: {tmd_test_auroc:.3f}")
        print(f"  Bootstrap Mean AUROC: {tmd_bootstrap['mean_auroc']:.3f} ± {tmd_bootstrap['std_auroc']:.3f}")
        print(f"  95% CI: [{tmd_bootstrap['ci_lower']:.3f}, {tmd_bootstrap['ci_upper']:.3f}]")
        print(f"  Valid bootstrap samples: {tmd_bootstrap['n_valid_samples']}/500")
    else:
        print("Error: Could not compute bootstrap confidence intervals")
