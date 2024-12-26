import pandas as pd

def extract_and_save_top_user_data(filepath, output_filepath):
    # Load the dataset
    col_names = ['user_id', 'timestamp', 'latitude', 'longitude', 'location_id']
    dataset = pd.read_csv(filepath, sep='\t', header=None, names=col_names)
    print(f"Dataset loaded with {len(dataset)} records.")

    # Count check-ins per user
    checkin_counts = dataset.groupby('user_id').size().reset_index(name='count')
    
    # Identify the user with the most check-ins
    max_checkins = checkin_counts['count'].max()
    top_users = checkin_counts[checkin_counts['count'] == max_checkins]
    
    # Extract this user's data
    top_users_data = dataset[dataset['user_id'].isin(top_users['user_id'])]
    
    # Save the top user's data to a text file
    top_users_data.to_csv(output_filepath, sep='\t', index=False, header=None)
    print(f"Data for user(s) with the most check-ins saved to {output_filepath}")

# Usage example:
filepath = './data/main/checkins-4sq.txt'  # Modify this with the actual file path
output_filepath = './1_user/user-4sq.txt'  # Modify this with your desired output file path
extract_and_save_top_user_data(filepath, output_filepath)
