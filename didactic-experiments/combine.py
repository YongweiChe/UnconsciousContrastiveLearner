# helper program to aggregate all experiments into a single CSV.

import pandas as pd

# Initialize an empty list to store the DataFrames
dfs = []

# Loop through the range of file indices
for i in range(20):  # Assuming the files are numbered from 0 to 19
    file_path = f'experiment_details{i}.csv'
    try: 
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Append the DataFrame to the list
        dfs.append(df)
    except:
        print(f'{file_path} does not exist')
# Concatenate all the DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv('combined_experiments.csv', index=False)

