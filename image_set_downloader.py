import os
import json

import cv2
from datetime import datetime

import tqdm
from roboflow import Roboflow
import multiprocessing as mp

from image_downloading import download_image

file_dir = os.path.dirname(__file__)
prefs_path = os.path.join(file_dir, 'preferences.json')

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="t7QUZv5sYTIvV50maVEf")

# Retrieve your current workspace and project name
print(rf.workspace())

workspaceId = 'steven-li'
projectId = 'rural-area-building'
project = rf.workspace(workspaceId).project(projectId)

zoom = 12
channels = 3
tile_size = 256
tag = "America"
number = 64
url = "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
headers = {
    "cache-control": "max-age=0",
    "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"99\", \"Google Chrome\";v=\"99\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36"
}

base_lat, base_lon = float("50.502062"), float("-125.961356")
increment = 1
batch_name = "_".join([tag, str(zoom), str(increment).replace(".", "_")])


def download(i):
    j = i // number
    i = i % number
    lat1 = base_lat - increment * (i + 0)
    lon1 = base_lon + increment * (j + 0)
    lat2 = base_lat - increment * (i + 1)
    lon2 = base_lon + increment * (j + 1)
    assert lat1 > lat2
    assert lon1 < lon2
    img = download_image(lat1, lon1, lat2, lon2, zoom, url,
                         headers, tile_size, channels)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = f'img_{timestamp}_{lat1}_{lon1}_{lat2}_{lon2}.png'
    cv2.imwrite(os.path.join("images", name), img)
    project.upload(os.path.join("images", name),
                   batch_name=batch_name,
                   tag=tag,
                   split="train",
                   num_retry_uploads=3)
    os.remove(os.path.join("images", name))
    tqdm.tqdm.write(f"Uploaded {name}")



def generator(number=32):
    return [i for i in range(number ** 2)]


def run():
    os.makedirs("images", exist_ok=True)
    for i in tqdm.tqdm(generator(number), total=number ** 2):
        download(i)


if __name__ == '__main__':
    run()
