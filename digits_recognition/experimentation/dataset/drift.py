import sys
import argparse

from alibi_detect.cd import KSDrift

from digits_recognition.experimentation.dataset.load_ubyte_data import load_ubyte_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str)
    parser.add_argument('--p_value', type=float)

    args = parser.parse_args()

    train_images, _, test_images, _ = load_ubyte_data(args.path)

    drift_detector = KSDrift(train_images / 255, p_val=args.p_value)

    prediction = drift_detector.predict(test_images / 255, return_p_val=True, return_distance=True)

    is_drift = prediction['data']['is_drift']

    if is_drift:
        print("Drift detected!")
        print(f'P-value: {prediction['data']['p_val'].mean()}')
        sys.exit(1)
    else:
        print("No drift detected.")
        sys.exit(0)