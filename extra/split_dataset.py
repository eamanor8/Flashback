import pandas as pd
from io import StringIO
from tqdm import tqdm
import time

# Paths to files
input_file = './data/checkins-4sq.txt'  # Path to the original file
batch1_output_file = './data/4sq-train-dataset.txt'
batch2_output_file = './data/4sq-test-dataset.txt'

start_time = time.time()  # Start the timer

# Load the dataset
print("Loading the dataset...")
with open(input_file, 'r') as infile:
    lines = infile.readlines()

df = pd.read_csv(StringIO(''.join(lines)), sep='\t', header=None, 
                 names=["user_id", "date-time", "latitude", "longitude", "location_id"])

# Get the unique user IDs
print("Extracting unique user IDs...")
unique_users = df['user_id'].unique()

# Ensure there are at least 10,000 unique users in the dataset
if len(unique_users) < 46065:
    raise ValueError("The dataset does not contain enough users.")

# Get the first and second batches of unique users
print("Splitting users into batches...")
batch1_users = unique_users[:23032]
batch2_users = unique_users[23032:46065]

# Filter out the data for the two batches of users
print("Filtering data for batch 1...")
batch1_users_data = df[df['user_id'].isin(tqdm(batch1_users, desc="Batch 1 Filtering"))]

print("Filtering data for batch 2...")
batch2_users_data = df[df['user_id'].isin(tqdm(batch2_users, desc="Batch 2 Filtering"))]

# Save the datasets
print("Saving batch 1 data...")
batch1_users_data.to_csv(batch1_output_file, sep='\t', header=False, index=False)

print("Saving batch 2 data...")
batch2_users_data.to_csv(batch2_output_file, sep='\t', header=False, index=False)

# Print summary and elapsed time
end_time = time.time()
print(f"Batch1: {len(batch1_users_data)} records saved to {batch1_output_file}")
print(f"Batch2: {len(batch2_users_data)} records saved to {batch2_output_file}")
print(f"Time taken: {end_time - start_time:.2f} seconds")
