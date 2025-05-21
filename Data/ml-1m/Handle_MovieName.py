import pandas as pd

# File paths
MOVIES_FILE = 'movies.dat'
GLOBAL_ID_FILE = 'globalId2ModelId.csv'
OUTPUT_FILE = 'modelId_name.csv'


def load_movies_data(file_path):
    """Load and preprocess the movies dataset."""
    columns = ['id', 'movie_name', 'genres']
    try:
        df = pd.read_csv(file_path, sep='::', engine='python', header=None, names=columns, encoding='latin-1')
        print("First 10 rows of movies data:")
        print(df.head(10))
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading movies data: {e}")
        return None


def load_global_id_mapping(file_path):
    """Load the global ID to model ID mapping."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading global ID mapping: {e}")
        return None


def merge_and_save_data(movies_df, global_id_df, output_file):
    """Merge movies and global ID data, then save the result."""
    if movies_df is None or global_id_df is None:
        print("Error: Cannot proceed with merging due to missing data.")
        return

    try:
        # Merge dataframes
        merged_df = pd.merge(movies_df, global_id_df, left_on='id', right_on='Global_ID', how='left')
        result_df = merged_df[['Model_ID', 'movie_name', 'genres']]

        # Save to CSV
        result_df.to_csv(output_file, index=False)
        print(f"Successfully saved merged data to {output_file}")
    except Exception as e:
        print(f"Error during merging or saving: {e}")


def main():
    """Main function to process movies data and merge with ID mapping."""
    # Load data
    movies_df = load_movies_data(MOVIES_FILE)
    global_id_df = load_global_id_mapping(GLOBAL_ID_FILE)

    # Merge and save
    merge_and_save_data(movies_df, global_id_df, OUTPUT_FILE)


if __name__ == "__main__":
    main()