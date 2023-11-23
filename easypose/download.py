import os
import requests

from tqdm import tqdm

from __version__ import __version__


ROOT_PATH = "~/.easypose"
ROOT_URL = "https://huggingface.co/dominic23331/easypose/resolve/main"
VERSION = __version__

os.environ["http_proxy"] = "http://127.0.0.1:4780"
os.environ["https_proxy"] = "http://127.0.0.1:4780"

def download(url, save_path, overwrite=True):
    save_path = os.path.expanduser(save_path)
    save_name = url.split("/")[-1]

    if overwrite or not os.path.exists(save_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fname = os.path.abspath(os.path.join(save_path, save_name))

        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)

        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                        r.iter_content(chunk_size=1024),
                        total=int(total_length / 1024.0 + 0.5),
                        unit='KB',
                        unit_scale=False,
                        dynamic_ncols=True,
                ):
                    f.write(chunk)


def download_model(model_file, detection_model=False, overwrite=True):
    model_info = model_file.split("_")
    model_name = model_info[0]
    model_type = model_info[1]
    model_type_dir = "detection" if detection_model else "pose"

    url = ROOT_URL + "/" + \
          model_type_dir + "/" + \
          VERSION + "/" + \
          model_name + "/" + \
          model_type + "/" + \
          model_file

    save_path = os.path.join(
        ROOT_PATH,
        model_type_dir,
        VERSION,
        model_name,
        model_type
    )

    download(url, save_path, overwrite)

if __name__ == '__main__':
    download_model("litehrnet_w18_heatmap_coco_256x192_20231009.onnx")

