import argparse
import os

import requests
import tqdm


FILES_TO_DOWNLOAD = {
    "policy_model": {
        "filename": "chembl_model.h5",
        "url": "https://ndownloader.figshare.com/files/23086454",
    }
}

def _download_file(url, filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(
            total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True
        )
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def main():
    parser = argparse.ArgumentParser("download_public_data")
    parser.add_argument(
        "path", default=".", help="the path download the files",
    )
    path = parser.parse_args().path

    try:
        for filespec in FILES_TO_DOWNLOAD.values():
            _download_file(filespec["url"], os.path.join(path, filespec["filename"]))
    except requests.HTTPError as err:
        print(f"Download failed with message {str(err)}")
        exit(1)

if __name__ == "__main__":
    main()