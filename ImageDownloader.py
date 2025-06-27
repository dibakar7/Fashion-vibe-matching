import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# to load CSV
image_df = pd.read_csv("images.csv")

# Output directory
os.makedirs("images", exist_ok=True)

# headers to mimic browser
headers = {"User-Agent": "Mozilla/5.0"}

# Keep track of saved filenames to avoid duplicates
existing_files = set(os.listdir("images"))

failure_log = open("failed_downloads.txt", "a")

for i, row in tqdm(image_df.iterrows(), total=len(image_df)):
    url = str(row['image_url']).strip()
    image_id = str(row['id'])

    filename = f"{image_id}_{i}.jpg"
    filepath = os.path.join("images", filename)

    if filename in existing_files:
        continue

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(filepath)
        print(f"File saved: {filename}")
    except Exception as e:
        print(f"Failed: {url} — {e}")
        print(f"Failed: {url} — {e}")
        failure_log.write(f"{image_id},{url}\n")

failure_log.close()
