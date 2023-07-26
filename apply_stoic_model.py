
from helper_functions import *
import numpy as np
import os
import pickle
import yaml


def stoic_severity_prediction(age, gender, dict_volumes, dict_intensities):
    path_current_dir = os.path.dirname(__file__)

    # Rescale kurtosis and skewness to [-1, 1]. Min, max values are taken from the training data
    with open(os.path.join(path_current_dir, 'rescale_values.yaml'), 'r') as stream:
        rescale_values = yaml.safe_load(stream)
    for feature in ['kurtosis_ggo', 'kurtosis_cons', 'kurtosis_healthy', 'skewness_ggo', 'skewness_cons', 'skewness_healthy']:
        dict_intensities[feature] = scale_values(dict_intensities[feature], rescale_values[f'{feature}_min'], rescale_values[f'{feature}_max'])

    # Create testing array
    X_test = np.array([[age, gender, dict_volumes['fraction_ggo'], dict_volumes['fraction_cons'],
                        dict_intensities['mean_ggo'], dict_intensities['mean_cons'], dict_intensities['mean_healthy'],
                        dict_intensities['kurtosis_ggo'], dict_intensities['kurtosis_cons'], dict_intensities['kurtosis_healthy'],
                        dict_intensities['skewness_ggo'], dict_intensities['skewness_cons'], dict_intensities['skewness_healthy']]])

    # Predict severity using the saved regression model
    probabilities_severe = []
    for bootstrap in range(20):
        model_severity = pickle.load(open(os.path.join(
            path_current_dir, 'models', 'regression_models', f'Severity_icolung_all_{bootstrap}'
        ), 'rb'))
        lr_prob_severe = model_severity.predict_proba(X_test)
        prob_severe = lr_prob_severe[:, 1][0]
        probabilities_severe.append(prob_severe)

    return np.mean(probabilities_severe)

