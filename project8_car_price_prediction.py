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


def generate_car_price_data(path: Path, rows: int = 2000):
    rng = np.random.default_rng(18)
    data = pd.DataFrame({
        'year': rng.integers(2000, 2023, rows),
        'mileage': rng.integers(1000, 180000, rows),
        'engine_size': rng.uniform(1.0, 4.5, rows),
        'horsepower': rng.integers(80, 300, rows),
        'brand': rng.integers(0, 6, rows),
        'fuel_type': rng.integers(0, 3, rows),
        'transmission': rng.integers(0, 2, rows),
    })

    age = 2026 - data['year']
    price = (
        18000
        + 2500 * (data['year'] - 2000)
        - 0.08 * data['mileage']
        + 3200 * data['engine_size']
        + 120 * data['horsepower']
        + 1500 * data['brand']
        + 1200 * data['fuel_type']
        + 800 * data['transmission']
        - 900 * age
    )
    data['price'] = np.round(price + rng.normal(0, 1800, rows), 2)
    data.to_csv(path, index=False)
    return data


def load_or_generate_data():
    file_path = DATA_DIR / 'car_prices.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return generate_car_price_data(file_path)


def main():
    df = load_or_generate_data()
    features = [col for col in df.columns if col != 'price']
    X = df[features]
    y = df['price']

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

    model_path = MODELS_DIR / 'car_price_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(search.best_estimator_, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
