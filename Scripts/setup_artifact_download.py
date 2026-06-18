# Copyright (c) Qualcomm Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import requests
from tqdm import tqdm
import os
import argparse

def download_large_file(url, output_path, chunk_size=1024*1024):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    filename = os.path.basename(output_path)

    with open(output_path, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a large file with progress bar.")
    parser.add_argument("-url", required=True, help="URL of the file to download")
    parser.add_argument("-outputfile", required=True, help="Full path (including filename) to save the downloaded file")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outputfile), exist_ok=True)
    download_large_file(args.url, args.outputfile)
