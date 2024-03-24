import pandas as pd

# Read the Excel file
df = pd.read_excel('./train/PNL_ADL_JIRACloud_1.xlsx')

# Write data to a CSV file
df.to_csv('./train/PNL_ADL_JIRACloud_1.csv', index=False)

print("CSV file has been created successfully.")
