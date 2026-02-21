import pandas as pd

# Load the CSV file
file_path = 'right.csv'
data = pd.read_csv(file_path)

# Select only the columns with 'Landmark_*_X' for modification
x_columns = [col for col in data.columns if 'Landmark' in col and '_X' in col]

# Apply the transformation: subtract 1 and multiply by -1
for col in x_columns:
    data[col] = (1 - data[col])

# Save the modified data to a new file
output_path = 'left.csv'
data.to_csv(output_path, index=False)

print(f"Modified file saved to {output_path}")
