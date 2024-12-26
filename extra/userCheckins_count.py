import pandas as pd

def load_dataset(filepath):
    """
    Load the dataset from a file.
    """
    col_names = ['user_id', 'timestamp', 'latitude', 'longitude', 'location_id']
    dataset = pd.read_csv(filepath, sep='\t', header=None, names=col_names)
    print(f"Dataset loaded with {len(dataset)} records.")

    # Assign index for each unique user_id
    dataset['index'], _ = pd.factorize(dataset['user_id'])

    # print maximum index and the user_id associated with it on one line
    print(f"Maximum Index: {dataset['index'].max()}, User ID: {dataset.loc[dataset['index'] == dataset['index'].max(), 'user_id'].iloc[0]}")

    #print total number of unique locations
    unique_locations = dataset['location_id'].nunique()
    print(f"Unique Locations: {unique_locations}")
    
    return dataset

def save_user_data(user_data, user_id, filename):
    """
    Save the data for a specific user to a text file, excluding the 'index' column.
    """
    # Drop the 'index' column from the DataFrame
    user_data = user_data.drop(columns=['index'])
    
    # Save data as a tab-separated text file
    user_data.to_csv(filename, sep='\t', index=False, header=None)
    print(f"Data for user ID {user_id} saved to {filename}")

def calculate_sequences(user_id, sequence_length, dataset):
    # Load data
    data = dataset

    
    # Filter data for the specified user ID
    user_data = data[data['index'] == user_id]

    # let's find the number of unique locations for the user
    unique_locations = user_data['location_id'].nunique()
    print(f"User ID: {user_id}, Unique Locations: {unique_locations}")
    
    # Calculate total check-ins
    total_checkins = len(user_data)
    
    # Calculate number of sequences using a sliding window
    if total_checkins < sequence_length:
        total_sequences = 0
    else:
        total_sequences = total_checkins - sequence_length + 1

    return user_data, total_checkins, total_sequences

# Example usage
user_id =  510  # Specify the user ID
sequence_length = 10  # Specify the sequence length
# dataset = load_dataset('./data/checkins-4sq.txt') # for test dataset
dataset = load_dataset('./data/main/checkins-4sq.txt') # for main dataset
filename = f'./single_user/user_data/user_{user_id}_data.txt'

# Get the indexed user_id
index_for_user_id = dataset.loc[dataset['user_id'] == user_id, 'index'].iloc[0] if not dataset[dataset['user_id'] == user_id].empty else None

# print original user_id and the index associated with it on one line
print(f"User ID: {user_id}, Index: {index_for_user_id}")

if index_for_user_id is not None:
    user_data, total_checkins, total_sequences = calculate_sequences(index_for_user_id, sequence_length, dataset)
    print(f"Total Check-ins: {total_checkins}")
    print(f"Total Sequences: {total_sequences}")

    # Save user data to a file
    save_user_data(user_data, user_id, filename)
else:
    print("User ID not found.")
