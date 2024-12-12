import pandas as pd

# Paths to input and output files
input_file = './data/main/checkins-gowalla.txt'  # Path to the original file
output_file = './data/checkins-gowalla.txt'  # Output file for saving the filtered users

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv(input_file, sep='\t', header=None, 
                 names=["user_id", "date-time", "latitude", "longitude", "location_id"])

# Step 1: Count the number of check-ins per user
print("Counting check-ins per user...")
user_checkin_counts = df['user_id'].value_counts()

# Step 2: Filter users with at least 101 check-ins
print("Filtering users with at least 101 check-ins...")
eligible_users = user_checkin_counts[user_checkin_counts >= 101].index

# Step 3: Select the first 45,344 eligible users
if len(eligible_users) < 45344:
    raise ValueError(f"There are only {len(eligible_users)} users with at least 101 check-ins.")
selected_users = eligible_users[:45344]

# Step 4: Filter the dataset for the selected users
print("Filtering dataset for selected users...")
filtered_df = df[df['user_id'].isin(selected_users)]

# Step 5: Save the filtered dataset to a file
print(f"Saving filtered dataset for {len(selected_users)} users...")
filtered_df.to_csv(output_file, sep='\t', header=False, index=False)

# Step 6: Calculate the total number of unique locations
print("Calculating total number of unique locations...")
unique_locations = filtered_df['location_id'].nunique()

# Print the results
print(f"Filtered dataset saved to {output_file}")
print(f"Total unique locations visited by the first 45,344 users with at least 101 check-ins: {unique_locations}")
