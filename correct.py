import pandas as pd

# Read the CSV file
df = pd.read_csv('CPP.csv', dtype={
    'crop': str,
    'month': str,
    'city': str,
    'state': str
})

# Display the first few rows to verify the changes
print(df.head())

# Check the data types of the columns
print(df.dtypes)

# Save the corrected data back to a CSV file
df.to_csv('crops_muj.csv', index=False)

print("Corrected CSV file has been saved as 'crops_muj.csv'")