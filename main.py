"""
COVID-19 severity prediction within one month.

"""

from apply_stoic_model import stoic_severity_prediction
from extract_intensities import intensities_and_textures
from extract_lesion_fractions import lesion_volumes_and_fractions
import glob
from google_drive_download import *
from helper_functions import *
from lung_lesion_segmentation import segment_lung_lesions
from natsort import humansorted
import os
import pandas as pd
import SimpleITK as sitk
from src_lung_lesions.data.study import convert_to_sitk
from src_lung_lesions.preprocessing.spatial import crop_around_mask
from src_lung_lesions.preprocessing.intensity import normalize_intensity


path_current_dir = os.path.dirname(__file__)
input_dir = os.path.join(path_current_dir, 'input')

# Download the models and test images from Google Drive
download_models(os.path.join(path_current_dir, 'models'))

# Segment the lungs and lesions for all patients in the input folder
segment_lung_lesions(
    input_dir,
    os.path.join(path_current_dir, 'models', 'covidAug21_multiclass_v0_split112', 'checkpoint_epoch=1400.pt')
)

# Extract features & run prediction
for path_scans in humansorted(glob.glob(os.path.join(input_dir, '*'))):
    if os.path.isdir(path_scans):
        patient_id = os.path.normpath(path_scans).split(os.path.sep)[-1]
        print(f'Running COVID-19 severity prediction for {patient_id}')

        # Read and preprocess images
        img = sitk.ReadImage(os.path.join(path_scans, 'CT.nii.gz'))
        img_np = sitk.GetArrayFromImage(img)
        lung_seg = sitk.ReadImage(os.path.join(path_scans, 'Lungs.nii.gz'))
        lung_seg_np = sitk.GetArrayFromImage(lung_seg)
        img_np = crop_around_mask(img_np, lung_seg_np, crop_margin=5)
        img_np = normalize_intensity(img_np)
        ct_preprocessed = convert_to_sitk(img_np, img)
        segmentation = sitk.ReadImage(os.path.join(path_scans, 'Lung_lesions.nii.gz'))

        # Lesion fractions
        dict_volumes = lesion_volumes_and_fractions(segmentation)

        # Intensities and textures
        dict_intensities = intensities_and_textures(ct_preprocessed, segmentation)

        # Age & gender
        df = pd.read_csv(os.path.join(input_dir, 'demographic_data.csv'), sep=';')
        age = df[df['Patient_id'] == patient_id]['Age'].values[0]
        gender = df[df['Patient_id'] == patient_id]['Gender'].values[0]
        age_cat = age_to_categorical(age)
        gender_cat = gender_to_categorical(gender)

        # Save the features
        with open('./features.csv', 'a') as csv_file:
            csv_file.write(
                f'{patient_id};{age_cat};{gender_cat};'
                f'{dict_volumes["volume_ggo"]};{dict_volumes["volume_cons"]};{dict_volumes["volume_lung"]};{dict_volumes["volume_total"]};'
                f'{dict_volumes["fraction_ggo"]};{dict_volumes["fraction_cons"]};{dict_volumes["fraction_lesion"]};'
                f'{dict_intensities["mean_healthy"]};{dict_intensities["kurtosis_healthy"]};{dict_intensities["skewness_healthy"]};'
                f'{dict_intensities["mean_ggo"]};{dict_intensities["kurtosis_ggo"]};{dict_intensities["skewness_ggo"]};'
                f'{dict_intensities["mean_cons"]};{dict_intensities["kurtosis_cons"]};{dict_intensities["skewness_cons"]}\n'
            )

        # Run prediction
        prob_severe = stoic_severity_prediction(age_cat, gender_cat, dict_volumes, dict_intensities)
        with open(os.path.join(input_dir, 'results.csv'), 'a+') as csv_file:
            csv_file.write(f'{patient_id};{prob_severe}\n')
