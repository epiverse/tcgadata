import requests
import zipfile
import io
import json
import csv
import os

def convert_json_embeddings_to_raw_tsv():
    """
    Downloads the tcga_titan_embeddings.json.zip file, extracts the JSON,
    and converts only the embeddings into a raw TSV file (no filenames or headers).
    """
    GITHUB_URL = "https://github.com/epiverse/tcgadata/raw/main/TITAN/tcga_titan_embeddings.json.zip"
    JSON_FILENAME = "tcga_titan_embeddings.json"
    OUTPUT_TSV_FILENAME = "tcga_titan_embeddings_raw.tsv"

    print(f"1. Downloading zip file from: {GITHUB_URL}")

    try:
        # Use requests to download the file content
        response = requests.get(GITHUB_URL, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Use io.BytesIO to treat the response content as a file-like object in memory
        zip_bytes = io.BytesIO(response.content)

        # 2. Open the zip file from the in-memory bytes
        with zipfile.ZipFile(zip_bytes, 'r') as zf:
            if JSON_FILENAME not in zf.namelist():
                print(f"Error: JSON file '{JSON_FILENAME}' not found inside the zip archive.")
                return

            # Read the JSON content
            print(f"2. Extracting and reading '{JSON_FILENAME}'...")
            with zf.open(JSON_FILENAME) as json_file:
                data = json.load(json_file)

        # 3. Process data
        embeddings = data.get("embeddings", [])

        if not embeddings:
            print("Error: 'embeddings' array is empty or missing in the JSON data.")
            return

        # Determine the embedding dimension
        embedding_dim = len(embeddings[0])
        num_entries = len(embeddings)
        print(f"Found {num_entries} embedding vectors, each with a {embedding_dim}-dimensional embedding.")

        # 4. Create the raw TSV file
        with open(OUTPUT_TSV_FILENAME, 'w', newline='', encoding='utf-8') as tsvfile:
            # Use the csv module with tab as the delimiter
            writer = csv.writer(tsvfile, delimiter='\t')

            # Write only the embedding rows
            print("5. Writing raw embedding data to TSV file...")
            for embedding in embeddings:
                # Ensure the embedding is converted to a list for the writerow function
                writer.writerow(list(embedding))

        print(f"\n✅ Successfully created {OUTPUT_TSV_FILENAME} with {num_entries} rows of embeddings.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")
    except json.JSONDecodeError:
        print("An error occurred: The extracted file is not a valid JSON document.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    convert_json_embeddings_to_raw_tsv()
