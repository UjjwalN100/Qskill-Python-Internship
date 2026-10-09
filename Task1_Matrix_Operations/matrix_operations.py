
import numpy as np


def input_matrix(name):
    """Read a matrix from the user's input."""
    print(f"\nEnter Matrix {name}")
    print("Enter each row as numbers separated by spaces.")

    while True:
        try:
            rows = int(input("Number of rows: "))
            cols = int(input("Number of columns: "))

            if rows <= 0 or cols <= 0:
                print("Rows and columns must be positive.")
                continue

            matrix = []

            for i in range(rows):
                while True:
                    values = input(
                        f"Enter row {i + 1}: "
                    ).split()

                    if len(values) != cols:
                        print(f"Please enter exactly {cols} values.")
                        continue

                    matrix.append([float(value) for value in values])
                    break

            return np.array(matrix)

        except ValueError:
            print("Invalid input. Please enter valid numbers.")


def display_matrix(matrix):
    """Display a matrix in a readable format."""
    print(np.array2string(matrix, precision=2, suppress_small=True))


def main():
    print("=" * 45)
    print("       MATRIX OPERATIONS TOOL")
    print("=" * 45)

    while True:
        print("\nChoose an operation:")
        print("1. Matrix Addition")
        print("2. Matrix Subtraction")
        print("3. Matrix Multiplication")
        print("4. Matrix Transpose")
        print("5. Matrix Determinant")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ").strip()

        if choice == "6":
            print("Thank you for using Matrix Operations Tool!")
            break

        if choice not in {"1", "2", "3", "4", "5"}:
            print("Invalid choice. Please select 1 to 6.")
            continue

        matrix_a = input_matrix("A")

        try:
            if choice in {"1", "2", "3"}:
                matrix_b = input_matrix("B")

                if choice in {"1", "2"}:
                    if matrix_a.shape != matrix_b.shape:
                        print(
                            "Error: Both matrices must have "
                            "the same dimensions."
                        )
                        continue

                    if choice == "1":
                        result = matrix_a + matrix_b
                        title = "Addition Result"
                    else:
                        result = matrix_a - matrix_b
                        title = "Subtraction Result"

                else:
                    if matrix_a.shape[1] != matrix_b.shape[0]:
                        print(
                            "Error: Columns of Matrix A must equal "
                            "rows of Matrix B."
                        )
                        continue

                    result = matrix_a @ matrix_b
                    title = "Multiplication Result"

            elif choice == "4":
                result = matrix_a.T
                title = "Transpose Result"

            else:
                if matrix_a.shape[0] != matrix_a.shape[1]:
                    print(
                        "Error: Determinant requires a square matrix."
                    )
                    continue

                result = np.linalg.det(matrix_a)
                title = "Determinant Result"

            print(f"\n{title}:")
            display_matrix(np.atleast_1d(result)) if np.isscalar(
                result
            ) else display_matrix(result)

            if choice == "5":
                print(f"Determinant value: {result:.4f}")

        except np.linalg.LinAlgError as error:
            print(f"Calculation error: {error}")

        except ValueError:
            print("Invalid numeric input. Please try again.")


if __name__ == "__main__":
    main()
