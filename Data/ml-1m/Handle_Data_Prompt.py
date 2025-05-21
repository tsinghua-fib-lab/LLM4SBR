import pandas as pd
import json
import csv
import pickle
from collections import Counter

# File paths
RATINGS_FILE = 'ratings.dat'
MOVIE_NAME_FILE = 'modelId_name.csv'
GLOBAL_ID_FILE = 'globalId2ModelId.csv'
TRAIN_FILE = 'Seq/train.txt'
TEST_FILE = 'Seq/test.txt'
ALL_TRAIN_SEQ_FILE = 'Seq/all_train_seq.txt'
TEST_JSON_FILE = 'Text/tes_session_long.json'

def load_and_preprocess_data(file_path):
    """Load and preprocess the ratings dataset."""
    column_names = ["UserId", "ItemId", "Rating", "Timestamp"]
    try:
        df = pd.read_csv(file_path, sep="::", engine="python", header=None, names=column_names)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df['Date'] = df['Timestamp'].dt.floor('10T')
        df['SessionID'] = df.groupby(['UserId', 'Date']).ngroup() + 1
        return df.sort_values(by=['SessionID', 'UserId', 'Date', 'Timestamp'])
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_sessions(data_df):
    """Create session sequences from the preprocessed DataFrame."""
    result_list = []
    grouped = data_df.groupby('SessionID')
    for session_id, group in grouped:
        sorted_group = group.sort_values(by='Timestamp')
        item_id_list = sorted_group['ItemId'].tolist()
        result_list.append(item_id_list)
    return result_list

def filter_sessions(result_list, min_count=5, min_length=2):
    """Filter items with less than min_count occurrences and sessions with less than min_length items."""
    flat_list = [item for sublist in result_list for item in sublist]
    count_dict = Counter(flat_list)
    filtered_nums = {num for num, count in count_dict.items() if count >= min_count}
    return [seq for seq in result_list if len(seq) >= min_length and all(num in filtered_nums for num in seq)]

def map_items_to_ids(result_list):
    """Map global item IDs to model IDs."""
    item_dict = {}
    item_ctr = 1
    handle_seq = []
    for seq in result_list:
        outseq = []
        for item in seq:
            item_str = str(item)
            if item_str in item_dict:
                outseq.append(item_dict[item_str])
            else:
                item_dict[item_str] = item_ctr
                outseq.append(item_ctr)
                item_ctr += 1
        handle_seq.append(outseq)
    return item_dict, handle_seq, item_ctr

def save_item_mapping(item_dict, output_file):
    """Save the item ID mapping to a CSV file."""
    df_id = pd.DataFrame(list(item_dict.items()), columns=['Global_ID', 'Model_ID'])
    df_id.to_csv(output_file, index=False)

def split_data(handle_seq, split_ratio=0.9):
    """Split sequences into training and test sets."""
    split_index = int(split_ratio * len(handle_seq))
    return handle_seq[:split_index], handle_seq[split_index:]

def count_clicks(train_seq, test_seq):
    """Count total clicks in train and test sequences."""
    clicks = sum(len(seq) for seq in train_seq) + sum(len(seq) for seq in test_seq)
    return clicks

def process_sequences(seq_list):
    """Process sequences to create input sequences and labels."""
    out_seqs = []
    labs = []
    for seq in seq_list:
        if len(seq) > 1:  # Ensure sequence has at least 2 items
            out_seqs.append(seq[:-1])
            labs.append(seq[-1])
    return out_seqs, labs

def compute_avg_sequence_length(train_seqs, test_seqs):
    """Compute the average sequence length."""
    total_length = sum(len(seq) for seq in train_seqs) + sum(len(seq) for seq in test_seqs)
    total_seqs = len(train_seqs) + len(test_seqs)
    return total_length / total_seqs if total_seqs > 0 else 0

def load_movie_names(file_path):
    """Load movie names from CSV and create ID-to-name mapping."""
    id2name_dict = {}
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)  # Skip header if present
            for row in csv_reader:
                if len(row) >= 3:
                    key, value_1, value_2 = row[0], row[1], row[2]
                    id2name_dict[key] = f"{value_1} -- {value_2}"
        return id2name_dict
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except Exception as e:
        print(f"Error loading movie names: {e}")
        return {}

def create_prompts(seq_list, id2name_dict):
    """Create text prompts for sequences."""
    seqs_text_list = []
    for seq in seq_list:
        seq_texts = "The order in which users click on items is as follows:\n"
        for i, item_id in enumerate(seq, 1):
            item_name = id2name_dict.get(str(item_id), str(item_id))
            seq_texts += f"{i}. {item_name}_{item_id}\n"
        seq_texts += "Please guess an item that the user is interested in in the long-term. (Only output the item name without any explanation.)"
        seqs_text_list.append({"prompt": seq_texts})
    return seqs_text_list

def save_data(data, file_path):
    """Save data to a pickle file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")

def main():
    # Load and preprocess data
    data_df = load_and_preprocess_data(RATINGS_FILE)
    if data_df is None:
        return

    # Create and filter sessions
    result_list = create_sessions(data_df)
    result_list = filter_sessions(result_list)
    print("Sample filtered sessions:", result_list[:5])

    # Map items to IDs
    item_dict, handle_seq, item_ctr = map_items_to_ids(result_list)
    print(f"Total unique items: {item_ctr}")
    print("Sample mapped sequences:", handle_seq[:5])

    # Save item mapping
    save_item_mapping(item_dict, GLOBAL_ID_FILE)

    # Save original sequences
    save_data(handle_seq, 'seq_origin.pkl')

    # Split data into train and test
    train_seq, test_seq = split_data(handle_seq)
    print("Sample training sequences:", train_seq[:5])
    print("Sample test sequences:", test_seq[:5])
    print(f"Training sequences: {len(train_seq)}")
    print(f"Test sequences: {len(test_seq)}")

    # Count total clicks
    total_clicks = count_clicks(train_seq, test_seq)
    print(f"Total clicks: {total_clicks}")

    # Process sequences for training and testing
    tra_seqs, tra_labs = process_sequences(train_seq)
    tes_seqs, tes_labs = process_sequences(test_seq)
    print("Sample training sequences:", tra_seqs[:10])
    print("Sample training labels:", tra_labs[:10])

    # Compute and print average sequence length
    avg_length = compute_avg_sequence_length(tra_seqs, tes_seqs)
    print(f"Average sequence length: {avg_length:.2f}")

    # Save processed sequences
    save_data((tra_seqs, tra_labs), TRAIN_FILE)
    save_data((tes_seqs, tes_labs), TEST_FILE)
    save_data(train_seq, ALL_TRAIN_SEQ_FILE)

    # Load movie names and create prompts
    id2name_dict = load_movie_names(MOVIE_NAME_FILE)
    seqs_text_list = create_prompts(tes_seqs, id2name_dict)

    # Save prompts to JSON
    try:
        with open(TEST_JSON_FILE, 'w', encoding='utf-8') as json_file:
            json.dump(seqs_text_list, json_file, indent=2)
    except Exception as e:
        print(f"Error saving JSON to {TEST_JSON_FILE}: {e}")

if __name__ == "__main__":
    main()
