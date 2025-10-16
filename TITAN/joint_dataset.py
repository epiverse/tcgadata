import requests
import zipfile
import io
import json
import numpy as np
import pandas as pd
import pprint

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
    # 1. Download and Load Data
    reports_data = load_zipped_json(URL_REPORTS)
    titan_data = load_zipped_json(URL_TITAN)

    # 2. Process TITAN Image Embeddings Data

    # Create a DataFrame for TITAN data for easier grouping and aggregation
    titan_df = pd.DataFrame({
        'filename': titan_data['filenames'],
        'embedding': titan_data['embeddings']
    })

    # Extract the patient_id (TCGA-XX-XXXX) from the filename.
    titan_df['patient_id'] = titan_df['filename'].str[:12]

    # 3. Aggregate TITAN Embeddings by Patient ID (Mean aggregation)
    print("Aggregating TITAN embeddings by patient_id (calculating mean embedding)...")

    # Calculate the mean of all image tile embeddings per patient
    aggregated_titan = titan_df.groupby('patient_id')['embedding'].apply(
        lambda x: np.mean(x.tolist(), axis=0).tolist()
    ).to_dict()

    # 4. Merge Aggregated Embeddings into Reports Data

    print("Merging data...")
    for report in reports_data:
        patient_id = report.get('patient_id')
        image_embedding = aggregated_titan.get(patient_id)
        report['image_embedding'] = image_embedding

    return reports_data

def save_to_tsv(merged_data, filename):
    """
    Converts the merged list of dictionaries to a DataFrame and saves it as a TSV.
    """
    print(f"\nConverting data to DataFrame and saving to {filename}...")

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(merged_data)

    # Select and reorder columns
    df = df[['i', 'id', 'text', 'embeddings', 'patient_id', 'cancer_type', 'image_embedding']]

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