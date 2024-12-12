import pandas as pd

# Paths to input and output files
input_file = './data/main/checkins-gowalla.txt'  # Path to the original file
output_file = './data/checkins-gowalla.txt'  # Output file path

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv(input_file, sep='\t', header=None, 
                 names=["user_id", "date-time", "latitude", "longitude", "location_id"])

# Step 1: Select the first 45,344 unique users
print("Selecting the first 45,344 unique users...")
unique_users = df['user_id'].unique()
selected_users = unique_users[:45344]  # Select the first 45,344 unique users

# Filter rows for the selected users
print("Filtering rows for selected users...")
user_filtered_df = df[df['user_id'].isin(selected_users)]

# Step 2: Check the number of unique locations
unique_locations = user_filtered_df['location_id'].unique()
num_locations = len(unique_locations)

if num_locations < 68881:
    raise ValueError(f"The total unique locations ({num_locations}) for the first 45,344 users is below 68,881.")
else:
    print(f"Found {num_locations} unique locations, selecting the first 68,881.")

# Step 3: Select the first 68,881 unique locations
selected_locations = unique_locations[:68881]  # Select the first 68,881 locations

# Filter rows for the selected users and locations
print("Filtering rows for selected users and selected locations...")
final_filtered_df = user_filtered_df[user_filtered_df['location_id'].isin(selected_locations)]

# Step 4: Save the subset to a file
print("Saving subset to file...")
final_filtered_df.to_csv(output_file, sep='\t', header=False, index=False)

# Print summary
print(f"Subset saved to {output_file}")
print(f"Total records: {len(final_filtered_df)}")
print(f"Unique Users: {final_filtered_df['user_id'].nunique()}")
print(f"Unique Locations: {final_filtered_df['location_id'].nunique()}")
