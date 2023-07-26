"""
Script to preprocess the COVID-19 lesions segmentation challenge data
"""


from src_lung_lesions.data.study import Study, get_patient_name_from_file_name, save, convert_to_sitk
from src_lung_lesions.preprocessing.segmentation import compute_lung_seg, add_lung_label
from src_lung_lesions.preprocessing.spatial import crop_around_mask, resample
from src_lung_lesions.preprocessing.intensity import normalize_intensity


def preprocess(study, patient_name=None):
    LABELS_BINARY = {
        'lesion': 1,
        'lung': 2,  # only non lesion
        'background': 0,
    }

    img_np = study.image
    seg_np = study.segmentation

    # Compute the lung mask automatically
    lung_seg_np = compute_lung_seg(study.image_sitk)
    lung_seg_sitk = convert_to_sitk(lung_seg_np, study.image_sitk)

    # Crop the volumes
    img_np = crop_around_mask(img_np, lung_seg_np, crop_margin=5)
    # lung_seg_np = crop_around_mask(lung_seg_np, lung_seg_np, crop_margin=5)  # Ine 16/02/2022: added

    # Normalize the image intensity
    img_np = normalize_intensity(img_np)

    img_sitk = convert_to_sitk(img_np, study.image_sitk)

    # Resample the image
    img_sitk = resample(img_sitk)
    # lung_seg_sitk = convert_to_sitk(lung_seg_np, study.image_sitk)
    # lung_seg_sitk = resample(lung_seg_sitk)

    if seg_np:  # Ine 16/02/2022: added this for data without existing segmentations
        # Add the other lung label
        seg_np = add_lung_label(seg_np, lung_seg_np, lung_label=LABELS_BINARY['lung'])

        # Crop the volumes
        seg_np = crop_around_mask(seg_np, lung_seg_np, crop_margin=5)

        # Resample the image and the segmentation
        seg_sitk = convert_to_sitk(seg_np, study.seg_sitk)
        seg_sitk = resample(seg_sitk, is_seg=True)
    else:
        seg_sitk = None

    return img_sitk, seg_sitk, lung_seg_sitk
