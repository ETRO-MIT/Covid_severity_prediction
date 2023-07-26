import os
import SimpleITK as sitk


def convert_to_sitk(img_np, ref_img_sitk):
    img_sitk = sitk.GetImageFromArray(img_np)
    img_sitk.SetOrigin(ref_img_sitk.GetOrigin())
    img_sitk.SetSpacing(ref_img_sitk.GetSpacing())
    img_sitk.SetDirection(ref_img_sitk.GetDirection())
    return img_sitk

def get_patient_name_from_file_name(file_name):
    name = file_name.replace('_ct.nii.gz', '').replace('_seg.nii.gz', '').replace('.nii.gz', '')
    return name

def save(img_sitk, seg_sitk, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # Save the image
    img_path = os.path.join(save_folder, 'ct.nii.gz')
    sitk.WriteImage(img_sitk, img_path)
    # Save the segmentation
    if seg_sitk is not None:
        seg_path = os.path.join(save_folder, 'seg.nii.gz')
        sitk.WriteImage(seg_sitk, seg_path)


class Study:
    def __init__(self, img_path, seg_path=None):
        self.img_path = img_path
        self.seg_path = seg_path
        self.name = get_patient_name_from_file_name(img_path)

    @property
    def image_sitk(self):
        return sitk.ReadImage(self.img_path)

    @property
    def image(self):
        img = sitk.GetArrayFromImage(self.image_sitk)
        return img

    @property
    def seg_sitk(self):
        if self.seg_path is not None:  # Ine 16/02/2022: added this for data without existing segmentations
            return sitk.ReadImage(self.seg_path)
        else:
            return None

    @property
    def segmentation(self):
        if self.seg_path is not None:  # Ine 16/02/2022: added this for data without existing segmentations
            seg = sitk.GetArrayFromImage(self.seg_sitk)
            return seg
        else:
            return None