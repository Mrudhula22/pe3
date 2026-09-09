from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_purchase_data(path: Path, rows: int = 1800):
    rng = np.random.default_rng(26)
    data = pd.DataFrame({
        'age': rng.integers(18, 70, rows),
        'income': rng.integers(20000, 150000, rows),
        'previous_purchases': rng.integers(0, 25, rows),
        'website_visits': rng.integers(1, 35, rows),
        'discount_used': rng.integers(0, 2, rows),
        'device_type': rng.integers(0, 3, rows),
        'campaign_clicked': rng.integers(0, 2, rows),
        'customer_segment': rng.integers(0, 5, rows),
    })

    purchase_score = (
        -5.0
        + 0.03 * data['age']
        + 0.00003 * data['income']
        + 0.30 * data['previous_purchases']
        + 0.15 * data['website_visits']
        + 0.85 * data['discount_used']
        + 0.45 * data['device_type']
        + 1.5 * data['campaign_clicked']
        + 0.6 * data['customer_segment']
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

    lr = LogisticRegression(max_iter=2000)
    dt = DecisionTreeClassifier(random_state=42, max_depth=6)
    rf = RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced')

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('dt', dt), ('rf', rf)],
        voting='soft'
    )

    ensemble.fit(X_train, y_train)

    pred = ensemble.predict(X_test)
    prob = ensemble.predict_proba(X_test)[:, 1]

    print('Cross-val scores (3-fold):')
    scores = cross_val_score(ensemble, X, y, cv=3, scoring='f1')
    print(scores)

    print('Accuracy:', round(accuracy_score(y_test, pred), 4))
    print('Precision:', round(precision_score(y_test, pred, zero_division=0), 4))
    print('Recall:', round(recall_score(y_test, pred, zero_division=0), 4))
    print('F1 Score:', round(f1_score(y_test, pred, zero_division=0), 4))
    print('ROC AUC:', round(roc_auc_score(y_test, prob), 4))

    model_path = MODELS_DIR / 'purchase_prediction_extended_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(ensemble, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
