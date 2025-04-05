import requests, random, os
import argparse
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download random recordings from Xeno-canto.')
parser.add_argument('--num_recordings', type=int, default=1000, 
                    help='Number of recordings to download (default: 1000)')
args = parser.parse_args()

# Set your output directory here
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xeno_mp3s')

print("Starting Xeno-canto download script...")

base_url = 'https://www.xeno-canto.org/api/2/recordings'
# Start with a simple query for any bird recording
query = 'q:A'  # Just request high quality recordings

# get initial data to find out how many pages there are
print("Fetching total number of pages...")
resp = requests.get(base_url, params={'query': query, 'page': 1})
resp.raise_for_status()  # Check for HTTP errors
data = resp.json()

# Print more details about the API response for debugging
print(f"API response: {data.keys()}")
print(f"Number of recordings in first page: {len(data.get('recordings', []))}")

num_pages = int(data.get('numPages', 1))
num_recordings = int(data.get('numRecordings', 0))
print(f"Found {num_pages} total pages with {num_recordings} recordings")

# Exit early if there are no recordings
if num_recordings == 0:
    print("No recordings found. Try adjusting the query parameters.")
    exit(1)

recordings = {}
# keep sampling random pages until we have the requested number of unique recordings
# or until we've tried a reasonable number of times
max_attempts = 100
attempts = 0
target_recordings = min(args.num_recordings, num_recordings)  # Don't try to get more than exist

pbar = tqdm(total=target_recordings, desc="Collecting recordings")
while len(recordings) < target_recordings and attempts < max_attempts:
    attempts += 1
    page = random.randint(1, num_pages)
    
    try:
        resp = requests.get(base_url, params={'query': query, 'page': page})
        resp.raise_for_status()
        data = resp.json()
        new_recordings = 0
        for rec in data.get('recordings', []):
            if rec['id'] not in recordings:
                recordings[rec['id']] = rec
                new_recordings += 1
        pbar.update(new_recordings)
    except Exception as e:
        print(f"Error fetching page {page}: {str(e)}")

# slice to the requested number of recordings
selected = list(recordings.values())[:target_recordings]
print(f"\nCollected {len(selected)} unique recordings")

# create a directory to store the files
os.makedirs(output_dir, exist_ok=True)
print(f"Saving files to: {os.path.abspath(output_dir)}")

# download each mp3 file
for rec in tqdm(selected, desc="Downloading MP3s"):
    mp3_url = rec.get('file')
    if mp3_url:
        file_name = os.path.join(output_dir, f"{rec['id']}.mp3")
        try:
            with requests.get(mp3_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(file_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"Error downloading {rec['id']}: {str(e)}")

print("\nDownload complete! 🎉")