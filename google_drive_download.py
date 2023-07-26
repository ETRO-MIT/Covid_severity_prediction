
import os
import gdown


def download_models(path_models):
    os.makedirs(os.path.join(path_models, 'covidAug21_multiclass_v0_split112'), exist_ok=True)
    os.makedirs(os.path.join(path_models, 'regression_models'), exist_ok=True)
    if not os.path.isfile(os.path.join(path_models, 'regression_models', 'Severity_icolung_all_19')):
        url = 'https://drive.google.com/drive/folders/1EJHYE3EOilh60d1752DpyG523UzJZ5Ly?usp=share_link'
        gdown.download_folder(url, output=os.path.join(path_models, 'regression_models'), quiet=False)

    # This file is too large to use gdown.download_folder
    if not os.path.isfile(
            os.path.join(path_models, 'covidAug21_multiclass_v0_split112', 'checkpoint_epoch=1400.pt')):
        url_file = 'https://drive.google.com/uc?id=1u85ED11XRFgU-Lj2PrYMmQtn1l-EGqQq&confirm=t'
        gdown.download(
            url_file,
            os.path.join(path_models, 'covidAug21_multiclass_v0_split112', 'checkpoint_epoch=1400.pt'),
            quiet=False
        )
