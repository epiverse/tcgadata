import requests
import zipfile
import io
import json
import numpy as np
import pandas as pd
import pprint
import os

# --- Configuration ---
# URLs for the datasets
URL_REPORTS = "https://raw.githubusercontent.com/epiverse/pathembed/main/tcgaPathReports.json.zip"
URL_TITAN = "https://raw.githubusercontent.com/epiverse/tcgadata/main/TITAN/tcga_titan_embeddings.json.zip"
OUTPUT_FILENAME = 'text_image_embeddings.tsv'
# ---------------------

def load_zipped_json(url):
    """Downloads a zipped JSON file from a URL and loads its content."""
    print(f"Downloading data from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    # Read the zip file content from memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Assuming there is only one JSON file in the zip
        file_name = z.namelist()[0]
        print(f"Loading file: {file_name}")
        with z.open(file_name) as f:
            data = json.load(f)
    return data

def process_and_merge_data():
    """
    Loads TCGA report and TITAN embedding data, aggregates TITAN embeddings by
    patient_id, and merges the result into the report data.
    """
    # 1. Download and Load Reports data
    reports_data = load_zipped_json(URL_REPORTS)

    # 2. Load TITAN image embeddings from local TSV produced by embeddingstsv.py
    titan_tsv_path = os.path.join(os.path.dirname(__file__), 'tcga_titan_embeddings_with_ids.tsv')
    if not os.path.exists(titan_tsv_path):
        raise FileNotFoundError(f"Required file not found: {titan_tsv_path}\nPlease run TITAN/embeddingstsv.py first to generate this TSV.")

    print(f"Loading TITAN embeddings from local TSV: {titan_tsv_path}")
    # Read TSV; embedding column contains a JSON string representing the list
    titan_df = pd.read_csv(titan_tsv_path, sep='\t', dtype={'filename': str, 'patient_id': str, 'embedding': str})

    # Ensure expected columns
    if 'filename' not in titan_df.columns or 'patient_id' not in titan_df.columns or 'embedding' not in titan_df.columns:
        raise ValueError(f"TITAN TSV must contain 'filename','patient_id','embedding' columns. Found: {list(titan_df.columns)}")

    # Keep only tiles that include 'DX1' in the filename
    titan_df['is_dx1'] = titan_df['filename'].astype(str).str.contains('DX1')
    titan_df_dx1 = titan_df[titan_df['is_dx1']].copy()

    print(f"Found {len(titan_df)} total TITAN tiles, {len(titan_df_dx1)} tiles with 'DX1'.")

    # Convert embedding JSON strings to lists
    titan_df_dx1['embedding_list'] = titan_df_dx1['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # Extract patient_id already present in the TSV but ensure it's a string
    titan_df_dx1['patient_id'] = titan_df_dx1['patient_id'].astype(str)

    # 3. Aggregate TITAN Embeddings by Patient ID using only DX1 tiles
    print("Aggregating TITAN DX1 embeddings by patient_id (calculating mean embedding)...")

    # Group and collect embedding lists and filenames per patient
    grouped = titan_df_dx1.groupby('patient_id').agg({
        'embedding_list': lambda s: list(s),
        'filename': lambda s: list(s)
    })

    # Build aggregated dicts: mean embedding and representative filename (first DX1 tile)
    aggregated_titan = {}
    aggregated_filenames = {}
    for pid, row in grouped.iterrows():
        lists = row['embedding_list']
        # compute mean across tile embeddings
        mean_emb = np.mean(lists, axis=0).tolist()
        aggregated_titan[pid] = mean_emb
        # choose a representative filename (first one)
        fnames = row['filename']
        rep = fnames[0] if fnames else ''
        aggregated_filenames[pid] = rep

    print(f"Aggregated embeddings for {len(aggregated_titan)} patients (only those with DX1 tiles).")

    # 4. Merge Aggregated Embeddings into Reports Data

    # 4. Merge Aggregated Embeddings into Reports Data
    print("Filtering reports to patients that have DX1 image embeddings and merging data...")

    # Keep only reports whose patient_id has an aggregated image embedding
    filtered_reports = []
    for report in reports_data:
        patient_id = report.get('patient_id')
        if patient_id in aggregated_titan:
            report['image_embedding'] = aggregated_titan.get(patient_id)
            # include representative DX1 filename
            report['image_filename'] = aggregated_filenames.get(patient_id, '')
            filtered_reports.append(report)

    print(f"Reports before filtering: {len(reports_data)}; after filtering: {len(filtered_reports)}")
    return filtered_reports

def save_to_tsv(merged_data, filename):
    """
    Converts the merged list of dictionaries to a DataFrame and saves it as a TSV.
    """
    print(f"\nConverting data to DataFrame and saving to {filename}...")

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(merged_data)

    # Select and reorder columns (include image_filename)
    # image_filename was added earlier when merging; include it in the output
    cols = ['i', 'id', 'text', 'embeddings', 'patient_id', 'cancer_type', 'image_filename', 'image_embedding']
    # Some older runs may not have image_filename present for every row; guard against missing column
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]

    # Convert list columns (embeddings) to JSON strings for proper TSV formatting
    for col in ['embeddings', 'image_embedding']:
        df[col] = df[col].apply(lambda x: json.dumps(x))

    # Save to TSV (tab-separated values)
    df.to_csv(filename, sep='\t', index=False)

    print("Saving complete.")
    print(f"Output file size (first 5 lines):")
    # Display the head of the file to verify
    with open(filename, 'r') as f:
        for i in range(5):
            print(f.readline().strip())


# --- Execution ---
if __name__ == '__main__':
    try:
        # Run the processing and merging logic
        merged_data = process_and_merge_data()

        # Save the final result as a TSV file
        save_to_tsv(merged_data, OUTPUT_FILENAME)

    except requests.exceptions.ConnectionError as e:
        print(f"\nERROR: Failed to download one or more files. Please check your network connection or the URLs.")
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")