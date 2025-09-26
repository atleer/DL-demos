import hashlib
import requests
import numpy as np
import os

def download_data(filenames):
  urls = ["https://osf.io/z3h78/download",
          "https://osf.io/ft5p3/download"]
  expected_md5s = ["85e1fe2ee8d936c1083d62563d79d958",
                  "e8f789abe20a7efde806d9ba03d20fd7"]

  for fname, url, expected_md5 in zip(filenames, urls, expected_md5s):
    if not os.path.exists(fname):
      try:
        r = requests.get(url)
      except requests.ConnectionError:
        print("Failed to download data")
      else:
        if r.status_code != requests.codes.ok:
          print("Failed to download data")
        elif hashlib.md5(r.content).hexdigest() != expected_md5:
          print("Data download appears corrupted")
        else:
          fname.parent.mkdir(parents=True, exist_ok=True)
          with open(fname, "wb") as fid:
            fid.write(r.content)