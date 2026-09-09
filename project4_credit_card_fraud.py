from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_fraud_data(path: Path, rows: int = 1500):
    rng = np.random.default_rng(24)
    data = pd.DataFrame({
        'transaction_amount': rng.uniform(1, 1500, rows),
        'merchant_risk': rng.integers(1, 6, rows),
        'distance_from_home': rng.uniform(0, 200, rows),
        'transaction_hour': rng.integers(0, 24, rows),
        'is_weekend': rng.integers(0, 2, rows),
        'device_changed': rng.integers(0, 2, rows),
        'country_mismatch': rng.integers(0, 2, rows),
    })

    fraud_score = (
        -3.0
        + 0.004 * data['transaction_amount']
        + 0.7 * data['merchant_risk']
        + 0.02 * data['distance_from_home']
        + 0.1 * data['transaction_hour']
        + 0.8 * data['is_weekend']
        + 1.2 * data['device_changed']
        + 1.5 * data['country_mismatch']
    )
    prob = 1 / (1 + np.exp(-fraud_score))
    data['is_fraud'] = (rng.random(rows) < (prob * 0.3)).astype(int)
    data.to_csv(path, index=False)
    return data


def load_or_generate_data():
    file_path = DATA_DIR / 'creditcard.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return generate_fraud_data(file_path)


def main():
    df = load_or_generate_data()
    features = [col for col in df.columns if col != 'is_fraud']
    X = df[features]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [6, 8, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    search = GridSearchCV(rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1')
    search.fit(X_train, y_train)

    pred = search.best_estimator_.predict(X_test)
    prob = search.best_estimator_.predict_proba(X_test)[:, 1]

    print('Best parameters:', search.best_params_)
    print('Accuracy:', round(accuracy_score(y_test, pred), 4))
    print('Precision:', round(precision_score(y_test, pred, zero_division=0), 4))
    print('Recall:', round(recall_score(y_test, pred, zero_division=0), 4))
    print('F1 Score:', round(f1_score(y_test, pred, zero_division=0), 4))
    print('ROC AUC:', round(roc_auc_score(y_test, prob), 4))

    model_path = MODELS_DIR / 'fraud_detection_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(search.best_estimator_, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
