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
        data['y_hr_' + y] = df['hr'].values

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

    print("Training HR model with grid search...")

    # Prepare features and labels for training
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']

    y_hr_train = data['y_hr_train']
    y_hr_val = data['y_hr_val']
    y_hr_test = data['y_hr_test']

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
    y_hr_train_val = np.concatenate([y_hr_train, y_hr_val])

    # Create validation indices for grid search (validation set indices in combined dataset)
    val_indices = list(range(len(X_train_scaled), len(X_train_val)))
    train_indices = list(range(len(X_train_scaled)))

    # Custom CV split for grid search (train on train set, validate on val set)
    custom_cv = [(train_indices, val_indices)]

    print("Optimizing HR model...")
    # Grid search for HR model
    ridge_hr = Ridge(random_state=42)
    grid_search_hr = GridSearchCV(
        ridge_hr,
        param_grid,
        cv=custom_cv,
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search_hr.fit(X_train_val, y_hr_train_val)

    print(f"\nBest HR parameters: {grid_search_hr.best_params_}")
    print(f"Best HR validation MAE: {-grid_search_hr.best_score_:.3f}")

    # Train final model on training set only with best parameters
    final_ridge_hr = Ridge(**grid_search_hr.best_params_, random_state=42)
    final_ridge_hr.fit(X_train_scaled, y_hr_train)

    # Make predictions on validation set
    hr_val_pred = final_ridge_hr.predict(X_val_scaled)

    # Make predictions on test set
    hr_test_pred = final_ridge_hr.predict(X_test_scaled)

    # Calculate validation metrics
    hr_val_mae = mean_absolute_error(y_hr_val, hr_val_pred)

    # Calculate test metrics
    hr_test_mae = mean_absolute_error(y_hr_test, hr_test_pred)

    print("\n=== Validation Results (with optimized hyperparameters) ===")
    print(f"HR - MAE: {hr_val_mae:.3f}")

    print("\n=== Test Results ===")
    print(f"HR - MAE: {hr_test_mae:.3f}")
