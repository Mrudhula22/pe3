from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def generate_house_price_data(path: Path, rows: int = 1800):
    rng = np.random.default_rng(8)
    data = pd.DataFrame({
        'sqft': rng.integers(600, 4500, rows),
        'bedrooms': rng.integers(1, 6, rows),
        'bathrooms': rng.integers(1, 4, rows),
        'age': rng.integers(0, 35, rows),
        'crime_rate': rng.uniform(0.1, 4.0, rows),
        'school_rating': rng.uniform(1, 10, rows),
    })

    price = (
        32000
        + 100 * data['sqft']
        + 25000 * data['bedrooms']
        + 22000 * data['bathrooms']
        - 900 * data['age']
        - 12000 * data['crime_rate']
        + 6000 * data['school_rating']
    )
    data['price'] = np.round(price + rng.normal(0, 12000, rows), 2)
    data.to_csv(path, index=False)
    return data


def load_or_generate_data():
    file_path = DATA_DIR / 'house_prices.csv'
    if file_path.exists():
        return pd.read_csv(file_path)
    return generate_house_price_data(file_path)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        'model': model,
        'r2': r2_score(y_test, pred),
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
    }


def main():
    df = load_or_generate_data()
    features = [col for col in df.columns if col != 'price']
    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.01, max_iter=10000)

    ridge_result = evaluate_model(ridge, X_train, X_test, y_train, y_test)
    lasso_result = evaluate_model(lasso, X_train, X_test, y_train, y_test)

    print('Ridge Results:')
    print('R2:', round(ridge_result['r2'], 4))
    print('MAE:', round(ridge_result['mae'], 4))
    print('RMSE:', round(ridge_result['rmse'], 4))

    print('\nLasso Results:')
    print('R2:', round(lasso_result['r2'], 4))
    print('MAE:', round(lasso_result['mae'], 4))
    print('RMSE:', round(lasso_result['rmse'], 4))

    model_path = MODELS_DIR / 'house_price_ridge_lasso_model.pkl'
    with open(model_path, 'wb') as f:
        import pickle
        pickle.dump({'ridge': ridge_result['model'], 'lasso': lasso_result['model']}, f)

    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
