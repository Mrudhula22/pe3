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


def generate_concrete_data(path: Path, rows: int = 2000):
    rng = np.random.default_rng(15)
    data = pd.DataFrame({
        'cement': rng.uniform(150, 500, rows),
        'blast_furnace_slag': rng.uniform(0, 250, rows),
        'fly_ash': rng.uniform(0, 200, rows),
        'water': rng.uniform(120, 220, rows),
        'superplasticizer': rng.uniform(0, 30, rows),
        'coarse_aggregate': rng.uniform(800, 1100, rows),
        'fine_aggregate': rng.uniform(500, 900, rows),
        'age': rng.integers(1, 91, rows),
    })

    strength = (
        22
        + 0.09 * data['cement']
        + 0.05 * data['blast_furnace_slag']
        + 0.04 * data['fly_ash']
        - 0.06 * data['water']
        + 0.12 * data['superplasticizer']
        + 0.01 * data['coarse_aggregate']
        + 0.01 * data['fine_aggregate']
        + 0.05 * data['age']
    )
    data['strength'] = np.round(strength + rng.normal(0, 5, rows), 2)
    data.to_csv(path, index=False)
    return data


def load_or_generate_data():
    file_path = DATA_DIR / 'concrete_strength.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return generate_concrete_data(file_path)


def main():
    df = load_or_generate_data()
    features = [col for col in df.columns if col != 'strength']
    X = df[features]
    y = df['strength']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    param_grid = {
        'n_estimators': [150, 250],
        'max_depth': [6, 8, None],
        'min_samples_leaf': [1, 2],
    }
    rf = RandomForestRegressor(random_state=42)
    search = GridSearchCV(rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
    search.fit(X_train, y_train)

    pred = search.best_estimator_.predict(X_test)
    print('Best parameters:', search.best_params_)
    print('R2 Score:', round(r2_score(y_test, pred), 4))
    print('MAE:', round(mean_absolute_error(y_test, pred), 4))
    print('RMSE:', round(np.sqrt(mean_squared_error(y_test, pred)), 4))

    model_path = MODELS_DIR / 'concrete_strength_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(search.best_estimator_, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
