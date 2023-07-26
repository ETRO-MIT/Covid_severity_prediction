# Copyright 2021 Lucas Fidon

"""
@brief  Script for performing segmentation inference
        for all the testing COVID-19 cases of Lucas
        using a CNN that was previously trained.
"""

import gc
import glob
import os
import time

import SimpleITK as sitk
import yaml
from src_lung_lesions.data.study import Study
from src_lung_lesions.inference import segment
from src_lung_lesions.preprocess import preprocess


def segment_lung_lesions(path_input, path_model):
    """
    Run inference for all the testing data
    """
    # Settings
    path_current_dir = os.path.dirname(__file__)
    path_config_file = os.path.join(path_current_dir, 'config_lung_lesions.yaml')
    with open(path_config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    with open(path_config_file, 'r') as stream:  # Use same file for everything (Lucas had it split up)
        data_config = yaml.safe_load(stream)
    nr_subdir = config['path_ids'].count('/')

    # Go over each CT
    for path_ct in sorted(glob.glob( os.path.join(path_input, f'{config["path_ids"]}{config["ct_filename"]}{config["file_extension"]}'))):
        case_name = '/'.join(path_ct.split('/')[-(nr_subdir + 1):-1])

        if os.path.isfile(os.path.join(os.path.split(path_ct)[0], 'Lung_lesions.nii.gz')):
            print('\n%s segmentation already exists. Skip inference.' % case_name)
            continue

        # Preprocess CT
        study = Study(img_path=path_ct, seg_path=None)  # no GT segmentation
        img_nii, seg_nii, lung_seg_sitk = preprocess(study=study)
        # sitk.WriteImage(img_nii, path_ct.replace(f'{config["ct_filename"]}{config["file_extension"]}',
        #                                          f'Lungs{config["file_extension"]}'))
        path_preprocessed_ct = path_ct.replace(
            f'{config["ct_filename"]}{config["file_extension"]}', f'CT_preprocessed{config["file_extension"]}'
        )
        sitk.WriteImage(img_nii, path_preprocessed_ct)

        # Segment the lesions
        input_path_dict = {'ct': path_preprocessed_ct}
        print(f'Segmenting the lung lesions for {path_ct}')
        t0 = time.time()
        segment(
            config=config,
            data_config=data_config,
            model_path=path_model,
            input_path_dict=input_path_dict,
            save_folder=os.path.join(path_input, case_name),
        )
        print('\tinference done in %.2f sec\n' % (time.time() - t0))

        # MONAI saves the prediction at save_folder/image_name/image_name.nii.gz
        # but we want to save them in the patient folder
        # so we have to move the files around...
        monai_save_path = os.path.join(path_input, case_name, 'CT_preprocessed', 'CT_preprocessed.nii.gz')
        save_path = os.path.join(os.path.split(path_ct)[0], 'Lung_lesions.nii.gz')
        os.system('cp %s %s' % (monai_save_path, save_path))
        os.system('rm -r %s' % os.path.join(path_input, case_name, 'CT_preprocessed'))

        gc.collect()  # collect garbage to avoid memory issues
