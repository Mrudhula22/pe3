from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_purchase_data(path: Path, rows: int = 1800):
    rng = np.random.default_rng(22)
    data = pd.DataFrame({
        'age': rng.integers(18, 70, rows),
        'income': rng.integers(20000, 150000, rows),
        'previous_purchases': rng.integers(0, 20, rows),
        'website_visits': rng.integers(1, 30, rows),
        'discount_used': rng.integers(0, 2, rows),
        'device_type': rng.integers(0, 3, rows),
        'campaign_clicked': rng.integers(0, 2, rows),
    })

    purchase_score = (
        -5.5
        + 0.03 * data['age']
        + 0.00003 * data['income']
        + 0.35 * data['previous_purchases']
        + 0.18 * data['website_visits']
        + 0.8 * data['discount_used']
        + 0.5 * data['device_type']
        + 1.6 * data['campaign_clicked']
    )
    prob = 1 / (1 + np.exp(-purchase_score))
    data['purchase'] = (rng.random(rows) < prob).astype(int)
    data.to_csv(path, index=False)
    return data


def load_or_generate_data():
    file_path = DATA_DIR / 'purchase_data.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return generate_purchase_data(file_path)


def main():
    df = load_or_generate_data()
    features = [col for col in df.columns if col != 'purchase']
    X = df[features]
    y = df['purchase']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    param_grid = {
        'n_estimators': [150, 250],
        'max_depth': [4, 6, 8],
        'min_samples_leaf': [1, 2],
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

    model_path = MODELS_DIR / 'purchase_prediction_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(search.best_estimator_, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
