from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_admission_data(path: Path, rows: int = 1200):
    rng = np.random.default_rng(13)
    data = pd.DataFrame({
        'GRE': rng.integers(260, 340, rows),
        'GPA': rng.uniform(2.0, 4.0, rows),
        'University_Rating': rng.integers(1, 5, rows),
        'SOP': rng.uniform(1.0, 5.0, rows),
        'LOR': rng.uniform(1.0, 5.0, rows),
        'Research': rng.integers(0, 2, rows),
    })

    score = (
        -13.5
        + 0.04 * data['GRE']
        + 2.20 * data['GPA']
        + 0.80 * data['University_Rating']
        + 0.60 * data['SOP']
        + 0.55 * data['LOR']
        + 0.70 * data['Research']
    )
    prob = 1 / (1 + np.exp(-score))
    data['Admit'] = (rng.random(rows) < prob).astype(int)
    data.to_csv(path, index=False)
    return data


def load_or_generate_data():
    file_path = DATA_DIR / 'student_admission.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return generate_admission_data(file_path)


def main():
    df = load_or_generate_data()
    features = [col for col in df.columns if col != 'Admit']
    X = df[features]
    y = df['Admit']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Baseline classification
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200],
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
    print('Precision:', round(precision_score(y_test, pred), 4))
    print('Recall:', round(recall_score(y_test, pred), 4))
    print('F1 Score:', round(f1_score(y_test, pred), 4))
    print('ROC AUC:', round(roc_auc_score(y_test, prob), 4))

    model_path = MODELS_DIR / 'student_admission_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(search.best_estimator_, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
