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


def generate_house_price_data(path: Path, rows: int = 1800):
    rng = np.random.default_rng(7)
    data = pd.DataFrame({
        'sqft': rng.integers(700, 5000, rows),
        'bedrooms': rng.integers(1, 6, rows),
        'bathrooms': rng.integers(1, 4, rows),
        'age': rng.integers(0, 40, rows),
        'distance_city_center': rng.uniform(1, 25, rows),
        'garage': rng.integers(0, 3, rows),
        'school_rating': rng.uniform(1, 10, rows),
        'crime_rate': rng.uniform(0.1, 3.0, rows),
    })

    price = (
        45000
        + 120 * data['sqft']
        + 23000 * data['bedrooms']
        + 18000 * data['bathrooms']
        - 1400 * data['age']
        - 5200 * data['distance_city_center']
        + 9500 * data['garage']
        + 7000 * data['school_rating']
        - 15000 * data['crime_rate']
    )
    data['price'] = np.round(price + rng.normal(0, 15000, rows), 2)
    data.to_csv(path, index=False)
    return data


def load_or_generate_data():
    file_path = DATA_DIR / 'house_prices.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return generate_house_price_data(file_path)


def main():
    df = load_or_generate_data()
    features = [col for col in df.columns if col != 'price']
    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    baseline = GradientBoostingRegressor(random_state=42)
    baseline.fit(X_train, y_train)

    param_grid = {
        'n_estimators': [150, 250],
        'max_depth': [6, 8, None],
        'min_samples_leaf': [1, 2],
    }
    rf = RandomForestRegressor(random_state=42)
    search = GridSearchCV(rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
    search.fit(X_train, y_train)

    preds = search.best_estimator_.predict(X_test)
    print('Best parameters:', search.best_params_)
    print('R2 Score:', round(r2_score(y_test, preds), 4))
    print('MAE:', round(mean_absolute_error(y_test, preds), 4))
    print('RMSE:', round(np.sqrt(mean_squared_error(y_test, preds)), 4))

    model_path = MODELS_DIR / 'house_price_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump(search.best_estimator_, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
