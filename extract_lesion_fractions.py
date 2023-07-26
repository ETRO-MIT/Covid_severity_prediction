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


def get_lesion_volume_per_lobe(segmentation, lobes, label):
    segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)
    lobes = sitk.Cast(lobes, sitk.sitkUInt8)
    lobes.CopyInformation(segmentation)

    volumes_lesion = []
    volumes_lobe = []
    threshold_filter = sitk.ThresholdImageFilter()
    mask_filter = sitk.MaskImageFilter()
    for lobe_nr in range(1, 6):
        threshold_filter.SetLower(lobe_nr)
        threshold_filter.SetUpper(lobe_nr)
        one_lobe = threshold_filter.Execute(lobes)

        # Get the lobe volume
        label_filter = sitk.LabelShapeStatisticsImageFilter()
        label_filter.Execute(one_lobe)
        labels = label_filter.GetLabels()
        if lobe_nr in labels:
            volume = label_filter.GetPhysicalSize(lobe_nr) / 1000  # GetPhysicalSize is in mm3 so /1000 to get ml
        else:
            volume = 0
        volumes_lobe.append(volume)

        # Get the volume of lesion in this lobe
        segmentation_one_lobe = mask_filter.Execute(segmentation, one_lobe)
        volumes_lesion.append(get_lesion_volume(segmentation_one_lobe, label))

    return volumes_lesion, volumes_lobe


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
