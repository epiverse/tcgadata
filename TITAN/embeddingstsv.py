import requests
import zipfile
import io
import json
import csv
import os
import sys


def convert_json_embeddings_to_tsv():
    """
    Loads `tcga_titan_embeddings.json.zip` (prefers a local file in the same
    directory as this script), validates embedding dimensions, and writes a TSV
    containing `filename`, `patient_id` and the `embedding` JSON string.

    Output file: tcga_titan_embeddings_with_ids.tsv
    """
    # Remote fallback URL (kept for compatibility)
    GITHUB_URL = "https://github.com/epiverse/tcgadata/raw/main/TITAN/tcga_titan_embeddings.json.zip"
    JSON_FILENAME = "tcga_titan_embeddings.json"
    OUTPUT_TSV_FILENAME = "tcga_titan_embeddings_with_ids.tsv"

    # Prefer local zip in the same directory as this script
    script_dir = os.path.dirname(__file__)
    local_zip_path = os.path.join(script_dir, 'tcga_titan_embeddings.json.zip')

    try:
        if os.path.exists(local_zip_path):
            print(f"Loading local file: {local_zip_path}")
            zf = zipfile.ZipFile(local_zip_path, 'r')
            use_stream = False
        else:
            print(f"Local file not found, downloading from: {GITHUB_URL}")
            resp = requests.get(GITHUB_URL, stream=True)
            resp.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(resp.content), 'r')
            use_stream = True

        with zf:
            if JSON_FILENAME not in zf.namelist():
                print(f"Error: JSON file '{JSON_FILENAME}' not found inside the zip archive.")
                return 1

            print(f"Extracting '{JSON_FILENAME}'...")
            with zf.open(JSON_FILENAME) as jf:
                data = json.load(jf)

        embeddings = data.get('embeddings')
        filenames = data.get('filenames') or data.get('file_names') or []

        if not embeddings:
            print("Error: 'embeddings' array is empty or missing in the JSON data.")
            return 1

        if filenames and len(filenames) != len(embeddings):
            print(f"Warning: filenames length ({len(filenames)}) != embeddings length ({len(embeddings)})")

        # Validate embedding dimension consistency
        lengths = [len(e) for e in embeddings]
        minl, maxl = min(lengths), max(lengths)
        print(f"Found {len(embeddings)} embeddings; min length={minl}, max length={maxl}")
        if minl != maxl:
            print("Error: embedding vectors have inconsistent dimensions; aborting.")
            return 1

        embedding_dim = minl
        if embedding_dim != 768:
            print(f"Warning: embedding dimension is {embedding_dim} (expected 768).")

        # Write TSV with filename, patient_id and embedding (as JSON string)
        print(f"Writing TSV to {OUTPUT_TSV_FILENAME} (includes filename, patient_id, embedding)...")
        with open(OUTPUT_TSV_FILENAME, 'w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out, delimiter='\t')
            # header
            writer.writerow(['filename', 'patient_id', 'embedding'])

            for idx, emb in enumerate(embeddings):
                fname = filenames[idx] if idx < len(filenames) else ''
                patient_id = fname[:12] if fname else ''
                # ensure it's a plain list
                emb_list = list(emb)
                writer.writerow([fname, patient_id, json.dumps(emb_list)])

        print("✅ TSV writing complete.")
        return 0

    except requests.exceptions.RequestException as e:
        print(f"Network/download error: {e}")
        return 1
    except zipfile.BadZipFile:
        print("Error: Bad zip file or unable to open archive.")
        return 1
    except json.JSONDecodeError:
        print("Error: extracted file is not valid JSON.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    rc = convert_json_embeddings_to_tsv()
    sys.exit(rc)
