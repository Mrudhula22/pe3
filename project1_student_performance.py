from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_student_performance_data(path: Path, rows: int = 1500):
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        'study_hours': rng.integers(1, 12, rows),
        'attendance': rng.integers(50, 100, rows),
        'sleep_hours': rng.uniform(4, 10, rows),
        'previous_score': rng.uniform(40, 100, rows),
        'practice_tests': rng.integers(0, 8, rows),
        'parental_support': rng.integers(0, 3, rows),
        'internet_access': rng.integers(0, 2, rows),
    })

    score = (
        25
        + 4.2 * data['study_hours']
        + 0.55 * data['attendance']
        + 2.8 * data['sleep_hours']
        + 0.65 * data['previous_score']
        + 4.5 * data['practice_tests']
        + 5.0 * data['parental_support']
        + 4.0 * data['internet_access']
    )
    data['score'] = np.clip(score + rng.normal(0, 5, rows), 0, 100)
    data.to_csv(path, index=False)
    return data


def load_or_generate_data():
    file_path = DATA_DIR / 'student_performance.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return generate_student_performance_data(file_path)


def main():
    df = load_or_generate_data()
    features = [col for col in df.columns if col != 'score']
    X = df[features]
    y = df['score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Baseline model
    baseline = GradientBoostingRegressor(random_state=42)
    baseline.fit(X_train, y_train)

    # Tuned model
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 8, None],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestRegressor(random_state=42)
    search = GridSearchCV(rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
    search.fit(X_train, y_train)

    predictions = search.best_estimator_.predict(X_test)
    print('Best parameters:', search.best_params_)
    print('R2 Score:', round(r2_score(y_test, predictions), 4))
    print('MAE:', round(mean_absolute_error(y_test, predictions), 4))
    print('RMSE:', round(np.sqrt(mean_squared_error(y_test, predictions)), 4))

    # Save model
    model_path = MODELS_DIR / 'student_performance_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(search.best_estimator_, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
