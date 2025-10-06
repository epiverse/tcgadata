import requests
import zipfile
import io
import json
import csv
import os

def extract_patient_id(filename):
    """
    Extracts the TCGA Participant ID (TCGA-XX-XXXX) from the long filename.
    The Participant ID is composed of the first three hyphen-separated segments.
    Example: 'TCGA-06-1087-01Z...' -> 'TCGA-06-1087'
    """
    parts = filename.split('-')
    # The Participant ID is the first three components: TCGA-XX-XXXX
    if parts and parts[0] == 'TCGA' and len(parts) >= 3:
        return '-'.join(parts[0:3])
    return None

def generate_filtered_tsvs():
    """
    Downloads source files, filters data based on matching patient IDs, and
    outputs two resulting TSV files (embeddings and cancer types).
    """
    EMBEDDING_ZIP_URL = "https://github.com/epiverse/tcgadata/raw/main/TITAN/tcga_titan_embeddings.json.zip"
    MAP_ZIP_URL = "https://github.com/epiverse/tcgadata/raw/main/tcgaPathReports.json.zip"
    MAP_JSON_FILENAME = "tcgaPathReports.json"
    EMBEDDING_JSON_FILENAME = "tcga_titan_embeddings.json"

    # Output file names
    OUTPUT_EMBEDDING_TSV = "tcga_filtered_embeddings.tsv"
    OUTPUT_CANCER_TYPE_TSV = "tcga_filtered_cancer_types.tsv"

    # --- 1. Create Patient ID to Cancer Type Map ---
    print("1. Downloading and processing patient map data...")
    patient_map = {}

    try:
        response_map_zip = requests.get(MAP_ZIP_URL, stream=True)
        response_map_zip.raise_for_status()
        zip_bytes = io.BytesIO(response_map_zip.content)

        with zipfile.ZipFile(zip_bytes, 'r') as zf:
            if MAP_JSON_FILENAME not in zf.namelist():
                print(f"Error: JSON file '{MAP_JSON_FILENAME}' not found inside the zip archive.")
                return

            with zf.open(MAP_JSON_FILENAME) as json_file:
                map_data = json.load(json_file)

                # The map data is a list of objects containing 'patient_id' and 'cancer_type'
                for item in map_data:
                    # Use 'patient_id' (e.g., "TCGA-BP-5195") directly as the key
                    pid = item.get('patient_id')
                    ctype = item.get('cancer_type')

                    if pid and ctype:
                        patient_map[pid] = ctype

        print(f"   Created map with {len(patient_map)} unique patient IDs from {len(map_data)} records.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading patient map zip: {e}")
        return
    except json.JSONDecodeError:
        print(f"Error: Extracted map file '{MAP_JSON_FILENAME}' is not valid JSON.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during map zip processing: {e}")
        return

    # --- 2. Get Embeddings and Filenames ---
    print("\n2. Downloading and extracting embedding data...")
    all_embeddings = []
    all_filenames = []
    try:
        response_zip = requests.get(EMBEDDING_ZIP_URL, stream=True)
        response_zip.raise_for_status()
        zip_bytes = io.BytesIO(response_zip.content)

        with zipfile.ZipFile(zip_bytes, 'r') as zf:
            with zf.open(EMBEDDING_JSON_FILENAME) as json_file:
                data = json.load(json_file)
                all_embeddings = data.get("embeddings", [])
                all_filenames = data.get("filenames", [])

        if not all_embeddings or not all_filenames or len(all_embeddings) != len(all_filenames):
            print("Error: Embedding or filename data is missing or mismatched.")
            return

        print(f"   Found {len(all_filenames)} total embedding vectors to process.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading embeddings zip: {e}")
        return
    except Exception as e:
        print(f"An error occurred during embedding processing: {e}")
        return

    # --- 3. Filter and Prepare Output Data ---
    print("\n3. Filtering data: matching filenames to cancer types...")
    filtered_embeddings = []
    filtered_cancer_types = []

    for filename, embedding in zip(all_filenames, all_embeddings):
        # Extract the 3-component participant ID from the TITAN filename
        participant_id = extract_patient_id(filename)

        # Check if the participant ID exists in the map
        cancer_type = patient_map.get(participant_id)

        if cancer_type:
            # Only include entries that successfully match
            filtered_embeddings.append(embedding)
            filtered_cancer_types.append([cancer_type])

    num_matched = len(filtered_embeddings)
    print(f"   Successfully matched and filtered {num_matched} entries.")

    if num_matched == 0:
        print("No matches found. Check ID formats.")
        return

    # --- 4. Write Two TSV Files ---
    print("\n4. Writing filtered data to two TSV files...")

    # A. Write Embeddings TSV
    try:
        with open(OUTPUT_EMBEDDING_TSV, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            for embedding in filtered_embeddings:
                # Ensure the embedding is converted to a list for csv.writer
                writer.writerow(list(embedding))
        print(f"✅ Successfully created {OUTPUT_EMBEDDING_TSV} with {num_matched} rows (raw embeddings).")
    except Exception as e:
        print(f"Error writing embedding TSV: {e}")

    # B. Write Cancer Types TSV (WITHOUT header)
    try:
        with open(OUTPUT_CANCER_TYPE_TSV, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            # Removed: writer.writerow(["cancer_type"])
            writer.writerows(filtered_cancer_types)
        print(f"✅ Successfully created {OUTPUT_CANCER_TYPE_TSV} with {num_matched} rows (cancer types, no header).")
    except Exception as e:
        print(f"Error writing cancer type TSV: {e}")


if __name__ == "__main__":
    generate_filtered_tsvs()
