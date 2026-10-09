
from pathlib import Path

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "house_price_model.joblib"


def main():
    if not MODEL_FILE.exists():
        print("Trained model not found.")
        print("Run train_model.py first.")
        return

    model = joblib.load(MODEL_FILE)

    print("=" * 45)
    print("       HOUSE PRICE PREDICTOR")
    print("=" * 45)
    print("Enter the house details below.")

    try:
        quality = int(input("Overall quality (1-10): "))
        area = float(input("Living area in square feet: "))
        basement = float(input("Basement area in square feet: "))
        garage = int(input("Garage capacity (cars): "))
        bathrooms = int(input("Number of full bathrooms: "))
        year = int(input("Year built: "))
        bedrooms = int(input("Number of bedrooms: "))
        neighborhood = input("Neighborhood (e.g. NAmes): ").strip()

        if not 1 <= quality <= 10:
            raise ValueError("Quality must be between 1 and 10.")

        if min(area, basement, garage, bathrooms, bedrooms) < 0:
            raise ValueError("House measurements cannot be negative.")

        if not 1800 <= year <= 2100:
            raise ValueError("Please enter a valid construction year.")

        if not neighborhood:
            raise ValueError("Neighborhood cannot be empty.")

        house = pd.DataFrame(
            [
                {
                    "OverallQual": quality,
                    "GrLivArea": area,
                    "TotalBsmtSF": basement,
                    "GarageCars": garage,
                    "FullBath": bathrooms,
                    "YearBuilt": year,
                    "BedroomAbvGr": bedrooms,
                    "Neighborhood": neighborhood,
                }
            ]
        )

        predicted_price = model.predict(house)[0]

        print("\nPREDICTION RESULT")
        print(f"Estimated sale price: ${predicted_price:,.2f}")
        print("This is a model estimate, not a guaranteed market price.")

    except ValueError as error:
        print(f"Invalid input: {error}")


if __name__ == "__main__":
    main()
