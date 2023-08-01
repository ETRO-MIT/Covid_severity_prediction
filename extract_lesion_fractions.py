"""

Labels
0: background
1: ground glass opacity
2: consolidation
3: lung

"""

import SimpleITK as sitk


def get_lesion_volume(segmentation, label):
    segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)

    label_filter = sitk.LabelShapeStatisticsImageFilter()
    label_filter.Execute(segmentation)
    labels = label_filter.GetLabels()

    # Can only calculate the volume if the label is present
    if label in labels:
        volume = label_filter.GetPhysicalSize(label) / 1000  # GetPhysicalSize is in mm3 so /1000 to get ml
    else:
        volume = 0

    return volume


def lesion_volumes_and_fractions(segmentation):
    volume_ggo = get_lesion_volume(segmentation, 1)
    volume_cons = get_lesion_volume(segmentation, 2)
    volume_lung = get_lesion_volume(segmentation, 3)    # healthy lung

    volume_total = volume_ggo + volume_cons + volume_lung

    fraction_ggo = volume_ggo / volume_total
    fraction_cons = volume_cons / volume_total
    fraction_lesion = (volume_ggo + volume_cons) / volume_total

    return {
        'volume_ggo': volume_ggo,
        'volume_cons': volume_cons,
        'volume_lung': volume_lung,
        'volume_total': volume_total,
        'fraction_ggo': fraction_ggo,
        'fraction_cons': fraction_cons,
        'fraction_lesion': fraction_lesion,
    }
