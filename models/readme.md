# Model Card: CNN Digit Classifier

## **Model Details**
- **Developer**: Gianmarco Turchiano
- **Model Date**: December 2024
- **Model Version**: v4.0
- **Model Type**: Convolutional Neural Network (CNN)

---

## **Intended Use**
- **Primary Intended Uses**: Classification of 28x28 pictures of handwritten digits (0-9).
- **Primary Intended Users**: Developers of applications that need to recognize handwritten inputs.
- **Out-of-Scope Use Cases**: 
  - Classification of multiple digits at a time (this is not an object recognition model). Such a use case requires pre-processing with low-level image processing methods.

---

## **Factors**
- **Relevant Factors**: 
  - The model performs excellently on grayscale, centered, and standardized 28x28 pixel images.
- **Evaluation Factors**: Accuracy, precision recall and F1-score on the test set.

---

## **Metrics**
- **Performance Metrics**: 
  - Test Accuracy: 99.35%.
- **Decision Thresholds**: Maximum softmax probability determines the predicted class.
- **Variation Approaches**: Performance on noisy, shifted, or scaled digits.

---

## **Evaluation Data**
- **Datasets**: MNIST dataset (10,000 test examples).
- **Motivation**: Benchmarking digit classifiers and showcasing CNN capabilities.

---

## **Training Data**
- **Details**: 
  - Training Data: 60,000 MNIST images split into training and validation sets.

---

## **Ethical Considerations**
- The dataset does not account for disabilities or cultural variations.