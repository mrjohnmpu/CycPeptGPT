import pandas as pd
import io

# Load data
df = pd.read_csv("./restored_validation.csv")

# Filter for 'linear' and 'head_to_tail'
filtered_df = df[df['Cyclization'].isin(['linear', 'head_to_tail'])]

# Save to new CSV
output_filename = 'filtered_peptides.csv'
filtered_df.to_csv(output_filename, index=False)

# Count occurrences
counts = filtered_df['Cyclization'].value_counts()

print("Filtered data saved to:", output_filename)
print("\nCounts:")
print(counts)