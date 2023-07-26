# Copyright 2021 Lucas Fidon
from src_lung_lesions.data.covid19_transform_pipeline import covid19_inference_transform
from src_lung_lesions.data.single_case_dataloader import single_case_dataloader

SUPPORTED_DATASET = [
    'COVID19',
]

SUPPORTED_DATA_AUGMENTATION_PIPELINES = [
    'nnUNet',
]


def get_single_case_dataloader(config, data_config, input_path_dict):
    """
    Typically used for inference.
    :param config:
    :param data_config:
    :param input_path_dict:
    :return:
    """
    dataset_name = data_config['name']

    # Check the dataset name
    assert dataset_name in SUPPORTED_DATASET, \
        'Found dataset %s. But only %s are supported for inference.' % \
        (dataset_name, str(SUPPORTED_DATASET))

    # Get the inference transform pipeline for the dataset
    if dataset_name == 'COVID19':
        inference_pipeline = covid19_inference_transform(
            config=config,
            image_keys=data_config['info']['image_keys'],
        )
    else:
        inference_pipeline = None
        NotImplementedError('Unknown dataset for inference transform: %s' % dataset_name)

    # Create the dataloader
    dataloader = single_case_dataloader(
        inference_transform=inference_pipeline,
        input_path_dict=input_path_dict,
    )

    return dataloader


