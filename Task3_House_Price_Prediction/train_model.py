
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "train.csv"
MODEL_FILE = PROJECT_DIR / "house_price_model.joblib"


def main():
    print("=" * 55)
    print("       HOUSE PRICE PREDICTION MODEL")
    print("=" * 55)

    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    if "SalePrice" not in df.columns:
        print("Error: The dataset must contain SalePrice.")
        return

    # Use a manageable set of relevant features.
    features = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "GarageCars",
        "FullBath",
        "YearBuilt",
        "BedroomAbvGr",
        "Neighborhood",
    ]

    df = df[features + ["SalePrice"]].copy()
    df = df.dropna(subset=["SalePrice"])

    X = df[features]
    y = df["SalePrice"]

    numeric_features = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "GarageCars",
        "FullBath",
        "YearBuilt",
        "BedroomAbvGr",
    ]
    categorical_features = ["Neighborhood"]

    # Fill missing numerical values with the median.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Fill missing categories and convert them to numeric indicators.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nTotal records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Features used: {len(features)}")

    print("\nTraining linear regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nMODEL EVALUATION")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")

    results = pd.DataFrame(
        {
            "Actual Price": y_test.iloc[:10].to_numpy(),
            "Predicted Price": predictions[:10],
        }
    )
    print("\nSAMPLE PREDICTIONS")
    print(results.round(2).to_string(index=False))

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")
    print("Training and evaluation completed successfully.")


if __name__ == "__main__":
    main()
