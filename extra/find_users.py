import pandas as pd

def load_unique_user_ids(filepath):
    """
    Load dataset and return a list of unique user IDs.
    """
    dataset = pd.read_csv(filepath, sep='\t', header=None, names=['user_id', 'timestamp', 'latitude', 'longitude', 'location_id'])
    unique_user_ids = dataset['user_id'].unique()
    return unique_user_ids

def filter_and_count_users(dataset_path, excluded_user_ids, min_checkins, required_number_of_locations):
    """
    Filter out users from the second dataset that are not in the excluded_user_ids,
    have at least min_checkins, and exactly required_number_of_locations.
    """
    dataset = pd.read_csv(dataset_path, sep='\t', header=None, names=['user_id', 'timestamp', 'latitude', 'longitude', 'location_id'])
    # Filter dataset by excluding user_ids that are in excluded_user_ids
    filtered_dataset = dataset[~dataset['user_id'].isin(excluded_user_ids)]
    # Group by user_id
    grouped = filtered_dataset.groupby('user_id')
    # Filter groups with at least min_checkins and exactly required_number_of_locations
    final_users = grouped.filter(lambda x: len(x) >= min_checkins and len(x['location_id'].unique()) == required_number_of_locations)
    # Get unique user_ids
    final_user_ids = final_users['user_id'].unique()
    return final_user_ids

def save_user_ids(user_ids, filename):
    """
    Save the final valid user IDs to a text file.
    """
    with open(filename, 'w') as file:
        for user_id in user_ids:
            file.write(f"{user_id}\n")
    print(f"User IDs saved to {filename}")

# Example usage
path_dataset1 = './data/checkins-4sq.txt'  # Path to the first dataset
path_dataset2 = './data/main/checkins-4sq.txt'  # Path to the second dataset
min_checkins = 50  # Minimum number of check-ins
required_number_of_locations = 28  # Required number of unique locations

# Load unique user IDs from the first dataset
user_ids_from_first = load_unique_user_ids(path_dataset1)

# Get valid user IDs from the second dataset
user_ids_not_in_first = filter_and_count_users(path_dataset2, user_ids_from_first, min_checkins, required_number_of_locations)

print(f"Unique user IDs not in the first dataset with at least {min_checkins} check-ins and exactly {required_number_of_locations} unique locations:")
print(user_ids_not_in_first)

# Save the final valid user IDs to a file
save_user_ids(user_ids_not_in_first, './single_user/users_not_in_test_dataset.txt')
