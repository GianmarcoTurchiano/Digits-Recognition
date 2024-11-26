"""
Scripts that checks expectations about the dataset.
"""
import sys
import argparse

import great_expectations as gx
import pandas as pd

from digits_recognition.experimentation.dataset.load_ubyte_data import load_ubyte_data


def _get_gx_batch(data_source, images, labels, asset_name):
    df = pd.DataFrame(
        {
            'image_shapes': [img.shape for img in images],
            'labels': labels
        }
    )

    asset = data_source.add_dataframe_asset(name=asset_name)
    batch_def = asset.add_batch_definition_whole_dataframe(f'{asset_name} batch')

    batch = batch_def.get_batch(batch_parameters={'dataframe': df})

    return batch


def _append_gx_result(batch, expectation, res):
    res.append(batch.validate(expectation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str)
    parser.add_argument('--image_width', type=int)
    parser.add_argument('--image_height', type=int)
    parser.add_argument('--classes', type=int)

    args = parser.parse_args()

    train_images, train_labels, test_images, test_labels = load_ubyte_data(args.path)

    context = gx.get_context()

    image_shape_expectation = gx.expectations.ExpectColumnValuesToBeInSet(
        column='image_shapes',
        value_set={(args.image_height, args.image_width)}
    )

    label_value_expectation = gx.expectations.ExpectColumnValuesToBeInSet(
        column='labels',
        value_set=range(args.classes)
    )

    data_source = context.data_sources.add_pandas('dataset')

    train_batch = _get_gx_batch(data_source, train_images, train_labels, 'train set')
    test_batch = _get_gx_batch(data_source, test_images, test_labels, 'test set')

    res = []

    _append_gx_result(train_batch, image_shape_expectation, res)
    _append_gx_result(train_batch, label_value_expectation, res)
    _append_gx_result(test_batch, image_shape_expectation, res)
    _append_gx_result(test_batch, label_value_expectation, res)

    for r in res:
        if not r.success:
            print(r)
            sys.exit(1)
