import pandas as pd

def filter_and_count_users(dataset_path, min_checkins, required_number_of_locations):
    """
    Load dataset and filter users who have at least min_checkins and exactly required_number_of_locations.
    """
    # Load dataset
    dataset = pd.read_csv(dataset_path, sep='\t', header=None, names=['user_id', 'timestamp', 'latitude', 'longitude', 'location_id'])

    # Group by user_id
    grouped = dataset.groupby('user_id')

    # Filter users based on conditions
    valid_users = grouped.filter(lambda x: len(x) >= min_checkins and len(x['location_id'].unique()) == required_number_of_locations)

    # Get unique user_ids
    valid_user_ids = valid_users['user_id'].unique()

    return valid_user_ids

def save_user_ids(user_ids, filename):
    """
    Save the final valid user IDs to a text file.
    """
    with open(filename, 'w') as file:
        for user_id in user_ids:
            file.write(f"{user_id}\n")
    print(f"User IDs saved to {filename}")

# Example usage
path_dataset = './data/main/checkins-4sq.txt'  # Path to the dataset
min_checkins = 50  # Minimum number of check-ins
required_number_of_locations = 21  # Required number of unique locations

# Get valid user IDs from the dataset
valid_user_ids = filter_and_count_users(path_dataset, min_checkins, required_number_of_locations)

print(f"User IDs with at least {min_checkins} check-ins and exactly {required_number_of_locations} unique locations:")
print(valid_user_ids)

# Save the final valid user IDs to a file
save_user_ids(valid_user_ids, './100_users/4sq-users-21-locations-.txt')