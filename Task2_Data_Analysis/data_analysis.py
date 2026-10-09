
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Allows charts to be saved without opening a window.

import matplotlib.pyplot as plt


# Locate the CSV file in the same folder as this Python script.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "dataset.csv"
CHART_DIR = PROJECT_DIR / "visualizations"


def main():
    print("=" * 55)
    print("       STUDENT PERFORMANCE DATA ANALYSIS")
    print("=" * 55)

    # 1. Load the CSV dataset.
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_FILE}")
        return

    if df.empty:
        print("Error: The dataset is empty.")
        return

    # 2. Display basic information.
    print("\n1. FIRST FIVE RECORDS")
    print(df.head().to_string(index=False))

    print("\n2. DATASET INFORMATION")
    print(f"Number of students: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # 3. Calculate statistics.
    numeric_df = df.select_dtypes(include="number")

    print("\n3. BASIC STATISTICS")
    print(numeric_df.describe().round(2).to_string())

    average_math = df["Math_Marks"].mean()
    average_science = df["Science_Marks"].mean()
    average_english = df["English_Marks"].mean()
    average_study_hours = df["Study_Hours"].mean()

    print("\n4. AVERAGE VALUES")
    print(f"Average mathematics marks: {average_math:.2f}")
    print(f"Average science marks: {average_science:.2f}")
    print(f"Average English marks: {average_english:.2f}")
    print(f"Average study hours: {average_study_hours:.2f}")

    # Create a folder for chart images.
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Bar chart: average marks by subject.
    subject_averages = df[
        ["Math_Marks", "Science_Marks", "English_Marks"]
    ].mean()

    plt.figure(figsize=(8, 5))
    subject_averages.plot(kind="bar")
    plt.title("Average Marks by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Average Marks")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "bar_chart.png", dpi=150)
    plt.close()

    # 6. Scatter plot: study hours vs mathematics marks.
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Study_Hours"], df["Math_Marks"])
    plt.title("Study Hours vs Mathematics Marks")
    plt.xlabel("Study Hours")
    plt.ylabel("Mathematics Marks")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "scatter_plot.png", dpi=150)
    plt.close()

    # 7. Heatmap: correlations between numerical columns.
    correlation = numeric_df.corr()

    plt.figure(figsize=(9, 7))
    image = plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )
    plt.colorbar(image, label="Correlation")
    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right"
    )
    plt.yticks(range(len(correlation.columns)), correlation.columns)

    for row in range(len(correlation.columns)):
        for col in range(len(correlation.columns)):
            plt.text(
                col,
                row,
                f"{correlation.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "heatmap.png", dpi=150)
    plt.close()

    # 8. Generate observations from the dataset.
    highest_subject = subject_averages.idxmax()
    lowest_subject = subject_averages.idxmin()
    correlation_value = df["Study_Hours"].corr(df["Math_Marks"])

    print("\n5. INSIGHTS AND OBSERVATIONS")
    print(
        f"- Highest average subject: {highest_subject} "
        f"({subject_averages.max():.2f} marks)."
    )
    print(
        f"- Lowest average subject: {lowest_subject} "
        f"({subject_averages.min():.2f} marks)."
    )
    print(
        f"- Correlation between study hours and mathematics marks: "
        f"{correlation_value:.2f}."
    )
    print(
        "- Correlation describes an association in this sample; "
        "it does not prove that study hours alone cause higher marks."
    )

    print("\n6. CHART FILES")
    print(f"Bar chart: {CHART_DIR / 'bar_chart.png'}")
    print(f"Scatter plot: {CHART_DIR / 'scatter_plot.png'}")
    print(f"Heatmap: {CHART_DIR / 'heatmap.png'}")

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
