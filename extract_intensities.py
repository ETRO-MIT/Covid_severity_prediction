"""
Labels
0: background
1: ground glass opacity
2: consolidation
3: lung

"""

import SimpleITK as sitk


def intensities_and_textures(img, lesions):
    lesions = sitk.Cast(lesions, sitk.sitkUInt8)

    labelstatsFilter = sitk.LabelIntensityStatisticsImageFilter()
    labelstatsFilter.Execute(lesions, img)
    labels = labelstatsFilter.GetLabels()
    if 3 in labels:
        mean_healthy = labelstatsFilter.GetMean(3)
        kurtosis_healthy = labelstatsFilter.GetKurtosis(3)
        skewness_healthy = labelstatsFilter.GetSkewness(3)
    else:
        mean_healthy = -1
        kurtosis_healthy = 6.1      # median value from the training data
        skewness_healthy = 2.3      # median value from the training data
    if 1 in labels:
        mean_ggo = labelstatsFilter.GetMean(1)
        kurtosis_ggo = labelstatsFilter.GetKurtosis(1)
        skewness_ggo = labelstatsFilter.GetSkewness(1)
    else:
        mean_ggo = -1               # healthy lung
        kurtosis_ggo = kurtosis_healthy
        skewness_ggo = skewness_healthy
    if 2 in labels:
        mean_cons = labelstatsFilter.GetMean(2)
        kurtosis_cons = labelstatsFilter.GetKurtosis(2)
        skewness_cons = labelstatsFilter.GetSkewness(2)
    else:
        mean_cons = -1
        kurtosis_cons = kurtosis_healthy
        skewness_cons = skewness_healthy


    return {
        'mean_healthy': mean_healthy,
        'kurtosis_healthy': kurtosis_healthy,
        'skewness_healthy': skewness_healthy,
        'mean_ggo': mean_ggo,
        'kurtosis_ggo': kurtosis_ggo,
        'skewness_ggo': skewness_ggo,
        'mean_cons': mean_cons,
        'kurtosis_cons': kurtosis_cons,
        'skewness_cons': skewness_cons,
    }
