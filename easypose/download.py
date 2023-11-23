import os
import requests

from tqdm import tqdm

from .consts import ROOT_PATH, ROOT_URL, VERSION


def download(url, save_path, overwrite=False):
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


def download_model(model_file, detection_model=False, overwrite=False):
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


