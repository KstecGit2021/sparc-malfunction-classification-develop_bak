::: content
[]{#markdown-mermaid aria-hidden="true" dark-mode-theme="dark"
light-mode-theme="default" max-text-size="50000"}

# Smart test predictor - User stories {#smart-test-predictor---user-stories .code-line line="0" dir="auto"}

## Types of users {#types-of-users .code-line line="2" dir="auto"}

1.  Standard users: Need to be able to easily upload datasets, initiate
    predictions and access basic analytics.

2.  Advanced users: Need to be able to customize model parameters and
    access advanced analytics.

## User stories {#user-stories .code-line line="8" dir="auto"}

### Submit dataset and generate predictions {#submit-dataset-and-generate-predictions .code-line line="10" dir="auto"}

As a standard user:

I want to upload my dataset with CP measurement data and generate
predictions

So that I can obtain the predicted FT outcome (pass/fail) for each chip

Acceptance criteria:

System accepts inference dataset by CSV upload

System accepts inference dataset by fetching from DB by
Inference_Dataset_ID

System generates pass/fail predictions for each chip (row in the
inference dataset)

Users can view predictions

Users can download predictions as CSV

Users can upload predictions to database

Priority: High

### Select features for prediction {#select-features-for-prediction .code-line line="29" dir="auto"}

As a standard user:

I want to include or exclude specific measurement columns from my
inference dataset

So that I can control which features are used for training and improve
prediction accuracy

Acceptance criteria:

System displays all detected columns with include/exclude options

System shows relevance scores for each column as guidance

System warns when selections significantly reduce training data

Users can save column selections for future use

Priority: High

### (Pre-training) Visualizing historical data & inference data for a given scenario {#pre-training-visualizing-historical-data--inference-data-for-a-given-scenario .code-line line="46" dir="auto"}

As a standard user:

I want to compare the distribution of historical data with the inference
dataset

So that I can identify potential gaps or biases in the training data
(E.g. data drift)

Acceptance criteria:

System provides visualizations (e.g., scatter plot, box plot) to compare
distributions

System highlights significant differences between historical and
inference data

Priority: Medium

### (Post-training) Visualize test results for a given scenario {#post-training-visualize-test-results-for-a-given-scenario .code-line line="61" dir="auto"}

As a standard user:

I want to visualize and rationalize the results of my tests

So that I can evaluate the quality and interpretability of the
predictions

Acceptance criteria:

System provides visualizations (e.g., confusion matrix, ROC curve) for
test results

System displays feature importance scores to explain predictions

Priority: Medium

### (Post-training) Compare 2 scenarios based on the same inference dataset with SHAP {#post-training-compare-2-scenarios-based-on-the-same-inference-dataset-with-shap .code-line line="76" dir="auto"}

As a standard user:

I want to analyze the impact of different features on model predictions
using SHAP values

So that I can gain insights into model behavior and improve feature
selection

Acceptance criteria:

System calculates SHAP values for each feature in the inference dataset

System provides visualizations (e.g., summary plots, dependence plots)
to interpret SHAP values

System allows comparison of SHAP values across different inference
scenarios

Priority: Medium

### Configure class imbalance handling {#configure-class-imbalance-handling .code-line line="92" dir="auto"}

As an advanced user:

I want to adjust oversampling techniques for the training dataset

So that I can address the imbalance between passing and failing chips to
improve model performance

Acceptance criteria:

System provides options for SMOTE, ADASYN, or no oversampling

System shows before/after class distribution when oversampling is
applied

System allows tuning of oversampling parameters

System validates that oversampling improves model performance metrics

System provides guidance on which technique works best for the current
dataset

Priority: Low

### Tune decision boundary for false negative reduction {#tune-decision-boundary-for-false-negative-reduction .code-line line="110" dir="auto"}

As an advanced user:

I want to adjust the classification threshold to reduce false negatives

So that I minimize the risk of shipping defective chips to customers

Acceptance criteria:

System provides a slider to adjust classification threshold (0.1 to 0.9)

System shows real-time preview of false negative/positive trade-offs

Priority: Low
:::
