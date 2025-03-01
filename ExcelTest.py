import pandas as pd


# Function to transfer Excel data to a 2D matrix
def excel_to_matrix(file_path, sheet_name=0):
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Convert DataFrame to a 2D matrix (list of lists)
    matrix = df.values.tolist()

    return matrix


# Example usage:
file_path = 'TestMap.xlsx'  # Replace with your file path
sheet_name = 0  # Replace with your sheet name or index if needed
matrix = excel_to_matrix(file_path, sheet_name)

# Print the 2D matrix
for row in matrix:
    print(row)
