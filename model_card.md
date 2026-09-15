# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classifier that predicts whether an individual earns more
than $50,000 per year based on U.S. Census demographic data. It was developed as
part of the Deploying a Scalable ML Pipeline with FastAPI project. The model is a
scikit-learn RandomForestClassifier trained with a fixed random_state=42 for
reproducibility, using default hyperparameters. Categorical features are encoded
using a one-hot encoder, and the target label is processed with a label binarizer.

## Intended Use

The model is intended to predict an individual's income category (>50K or
<=50K) from demographic attributes such as age, education, occupation, and
hours worked per week. It is designed for educational purposes to demonstrate a
deployable machine learning pipeline served through a FastAPI REST API. It is not
intended for use in real-world decisions affecting individuals, such as lending,
hiring, or credit determination.

## Training Data

The training data comes from the publicly available UCI Census Income dataset
(https://archive.ics.uci.edu/ml/datasets/census+income). The dataset contains
32,561 rows and 15 columns, including demographic and employment-related features
and a salary label. The data was split into 80% for training and 20% for testing
using a stratified split on the salary label to preserve the class distribution.
The eight categorical features were one-hot encoded, and the salary label was
binarized into 0 (<=50K) and 1 (>50K).

## Evaluation Data

The evaluation data consists of the 20% test split held out from the original
Census dataset. The same one-hot encoder and label binarizer fitted on the
training data were applied to the test data to ensure consistent feature
processing between training and evaluation.

## Metrics

The model was evaluated using precision, recall, and F1 score (F-beta with
beta=1). On the held-out test set, the model achieved the following performance:

- Precision: 0.7353
- Recall: 0.6378
- F1: 0.6831

In addition, performance was computed on slices of the data for each unique value
of every categorical feature. These per-slice metrics are saved in
slice_output.txt and show how model performance varies across subgroups such as
workclass, education, and occupation.

## Ethical Considerations

The Census dataset contains sensitive demographic attributes including race, sex,
and native country. Because the model is trained on historical data, it may
reflect and reproduce societal biases present in that data. Performance also
varies across data slices, meaning the model may be less accurate for certain
subgroups. This model should not be used to make consequential decisions about
individuals, and any real-world application would require careful fairness
auditing.

## Caveats and Recommendations

The model uses default RandomForest hyperparameters and was not tuned, so
performance could likely be improved through hyperparameter optimization or
cross-validation. Some categorical slices contain very few samples, which makes
their metrics unreliable. The dataset is also from 1994 and does not reflect
current economic conditions. Users should treat this model as a demonstration of
an ML deployment pipeline rather than a production-ready predictor.
