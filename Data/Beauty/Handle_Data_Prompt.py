import pickle
import csv
import json
import pandas as pd
from collections import Counter

# File paths
SESSIONS_FILE = 'sessions.csv'
MODEL_ID_NAME_FILE = 'modelId_name.csv'
TRAIN_FILE = '../train.txt'
TEST_FILE = '../test.txt'
ALL_TRAIN_SEQ_FILE = '../all_train_seq.txt'
OUTPUT_JSON_FILE = 'tra_session_long.json'

def load_sessions_data(file_path):
    """Load and preprocess the sessions dataset."""
    try:
        df = pd.read_csv(file_path)
        if 'SessionId' not in df.columns or 'ItemId' not in df.columns or 'Time' not in df.columns:
            raise ValueError("Required columns (SessionId, ItemId, Time) not found in CSV.")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading sessions data: {e}")
        return None

def create_sessions(df):
    """Create session sequences from the DataFrame."""
    result_list = []
    grouped = df.groupby('SessionId')
    for session_id, group in grouped:
        sorted_group = group.sort_values(by='Time')
        item_id_list = sorted_group['ItemId'].tolist()
        result_list.append(item_id_list)
    return result_list

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

def split_data(handle_seq, split_ratio=0.9):
    """Split sequences into training and test sets."""
    split_index = int(split_ratio * len(handle_seq))
    return handle_seq[:split_index], handle_seq[split_index:]

def count_clicks(train_seq, test_seq):
    """Count total clicks in train and test sequences."""
    return sum(len(seq) for seq in train_seq) + sum(len(seq) for seq in test_seq)

def process_sequences(seq):
    """Process sequences to create input sequences and labels for all possible suffixes."""
    out_seqs = []
    labs = []
    for i in range(1, len(seq)):
        labs.append(seq[-i])
        out_seqs.append(seq[:-i])
    return out_seqs, labs

def compute_avg_sequence_length(train_seqs, test_seqs):
    """Compute the average sequence length."""
    total_length = sum(len(seq) for seq in train_seqs) + sum(len(seq) for seq in test_seqs)
    total_seqs = len(train_seqs) + len(test_seqs)
    return total_length / total_seqs if total_seqs > 0 else 0

def load_id_to_name_mapping(file_path):
    """Load model ID to name mapping from CSV."""
    id2name_dict = {}
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None)  # Skip header if present
            for row in csv_reader:
                if len(row) >= 2:
                    id2name_dict[row[0]] = row[1]
        return id2name_dict
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except Exception as e:
        print(f"Error loading ID-to-name mapping: {e}")
        return {}

def create_prompts(seq_list, id2name_dict):
    """Create text prompts for sequences."""
    seqs_text_list = []
    for count, seq in enumerate(seq_list):
        seq_texts = "The order in which users click on items is as follows:\n"
        for i, item_id in enumerate(seq, 1):
            item_name = id2name_dict.get(str(item_id), str(item_id))
            if item_name == str(item_id):
                print(f"Warning: No name found for item ID {item_id} in sequence {count}: {seq}")
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

def save_json(data, file_path):
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2)
        print(f"JSON data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")

def main():
    """Main function to process sessions and generate prompts."""
    # Load sessions data
    df = load_sessions_data(SESSIONS_FILE)
    if df is None:
        return

    # Create session sequences
    result_list = create_sessions(df)
    print("Sample sessions:", result_list[:5])

    # Map items to IDs
    item_dict, handle_seq, item_ctr = map_items_to_ids(result_list)
    print(f"Total unique items: {item_ctr}")
    print("Sample mapped sequences:", handle_seq[:5])

    # Optional: Save original sequences
    # save_data(handle_seq, 'seq_origin.pkl')

    # Split data
    train_seq, test_seq = split_data(handle_seq)
    print("Sample training sequences:", train_seq[:5])
    print("Sample test sequences:", test_seq[:5])
    print(f"Training sequences: {len(train_seq)}")
    print(f"Test sequences: {len(test_seq)}")

    # Count clicks
    total_clicks = count_clicks(train_seq, test_seq)
    print(f"Total clicks: {total_clicks}")

    # Process sequences for training and testing
    tra_seqs, tra_labs = [], []
    for data in train_seq:
        seqs, labs = process_sequences(data)
        tra_seqs.extend(seqs)
        tra_labs.extend(labs)

    tes_seqs, tes_labs = [], []
    for data in test_seq:
        seqs, labs = process_sequences(data)
        tes_seqs.extend(seqs)
        tes_labs.extend(labs)

    print("Sample training sequences:", tra_seqs[:10])
    print("Sample training labels:", tra_labs[:10])
    print(f"Training sequences count: {len(tra_seqs)}")
    print(f"Test sequences count: {len(tes_seqs)}")

    # Compute average sequence length
    avg_length = compute_avg_sequence_length(tra_seqs, tes_seqs)
    print(f"Average sequence length: {avg_length:.2f}")

    # Optional: Save processed sequences
    save_data((tra_seqs, tra_labs), TRAIN_FILE)
    save_data((tes_seqs, tes_labs), TEST_FILE)
    save_data(train_seq, ALL_TRAIN_SEQ_FILE)

    # Load ID-to-name mapping and create prompts
    id2name_dict = load_id_to_name_mapping(MODEL_ID_NAME_FILE)
    seqs_text_list = create_prompts(tra_seqs, id2name_dict)
    print(f"Total prompts generated: {len(seqs_text_list)}")
    print("Sample prompts:", seqs_text_list[:5])

    # Save prompts to JSON
    save_json(seqs_text_list, OUTPUT_JSON_FILE)

if __name__ == "__main__":
    main()