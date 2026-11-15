# Comparative Evaluation of Six Models for Diabetes Risk Prediction

## Abstract

This report evaluates six candidate models for binary diabetes-risk prediction using the supplied Diabetes Prediction Dataset. The dataset contains **100,000 records**, eight predictors, and a diabetes label. The positive class represents 8,500 records (8.50%), so the study uses train-only SMOTENC to address imbalance while preserving a realistic untouched test set. The models are XGBoost, a deep neural network (DNN), multilayer perceptron (MLP), TabNet, Wide and Deep, and TabPFN. The strongest completed model by ROC-AUC was **XGBoost** with ROC-AUC 0.9780 and F1 0.8018. Metrics should be interpreted as an offline benchmark rather than clinical validation.

## 1 Introduction

Diabetes screening is a binary classification problem in which class imbalance makes accuracy alone an incomplete measure of performance. This project extends the supplied DNN prototype into a reproducible comparison of one gradient-boosting model and five neural or pretrained-tabular approaches. The practical objective is to identify a strong candidate model and document the full evaluation needed for a final-year research project.

The supplied reference is Iftikhar et al. (2025), *A Novel Explainable Deep Learning Framework for Accurate Diabetes Mellitus Prediction*, which reports an EchoceptionNet study on 100,000 records. The present work benchmarks the six models in the project architecture on the supplied CSV and does not claim to reproduce EchoceptionNet.

## 2 Dataset and preprocessing

The CSV contains 100,000 rows and 9 columns: four categorical/binary predictors (gender, hypertension, heart disease, and smoking history), four continuous predictors (age, BMI, HbA1c level, and blood glucose level), and the diabetes target. There are no missing values in the supplied file. The overall class counts are 91,500 negative and 8,500 positive examples.

The data were split once using stratification: 80,000 rows for training and 20,000 rows for testing. Encoders and scalers were fit on the training partition only. Categorical variables were ordinal-encoded for SMOTENC, then one-hot encoded; continuous variables were standardized. SMOTENC was applied only to the training partition, resulting in 146,400 balanced training rows. The test set remains imbalanced and untouched.

TabPFN was fit on a balanced 500-row subset of the balanced training data, as a documented computational constraint of the pretrained model. It was evaluated on the same test set as every other model.

## 3 Models

XGBoost uses histogram-based gradient-boosted trees. DNN is the existing three-hidden-layer feed-forward network. MLP is a scikit-learn feed-forward network with two hidden layers and early stopping. TabNet is an attentive tabular network. Wide and Deep combines a linear wide branch with a nonlinear deep branch. TabPFN is a pretrained tabular foundation model. The benchmark fixes a random seed of 42 where each backend supports it.

## 4 Evaluation protocol

Each model outputs a probability for the positive class. A threshold of 0.50 produces the reported class predictions. The primary metrics are accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC. Confusion-matrix counts, classification reports, predictions, ROC points, runtime metadata, model artifacts, and permutation-based feature-importance estimates are included in the `benchmark_outputs` directory. Because the positive class is only 8.50% of the data, recall, F1, ROC-AUC, and PR-AUC are more informative than accuracy alone.

## 5 Results

| Model         | Accuracy   | Precision   | Recall   | F1     | ROC-AUC   | PR-AUC   |
|:--------------|:-----------|:------------|:---------|:-------|:----------|:---------|
| XGBoost       | 96.98%     | 90.65%      | 71.88%   | 80.18% | 97.80%    | 88.09%   |
| Wide and Deep | 92.25%     | 52.64%      | 87.94%   | 65.86% | 97.60%    | 87.70%   |
| DNN           | 92.72%     | 54.53%      | 86.12%   | 66.77% | 97.57%    | 87.50%   |
| TabPFN        | 96.13%     | 78.13%      | 75.65%   | 76.87% | 97.22%    | 86.84%   |
| MLP           | 92.09%     | 52.14%      | 85.35%   | 64.73% | 97.18%    | 86.35%   |
| TabNet        | 82.30%     | 31.97%      | 96.00%   | 47.96% | 96.64%    | 83.48%   |

![Model comparison](benchmark_outputs/model_comparison.png)

![ROC curves](benchmark_outputs/roc_curves.png)

The best model by ROC-AUC was XGBoost (0.9780). Its accuracy was 0.9698, precision was 0.9065, recall was 0.7188, and F1-score was 0.8018. The full confusion matrix and class-specific precision/recall values should be read alongside these summary metrics, particularly when comparing models with different precision-recall trade-offs.

## 6 Interpretation and limitations

The benchmark is an offline comparison on one supplied dataset split. It does not establish generalization to a different hospital, population, device, or time period. The labels and features are dataset-level variables rather than a prospective clinical protocol. SMOTENC can improve minority-class learning, but synthetic examples do not replace external validation. TabPFN uses fewer training rows than the other models, so its result is a resource-constrained comparison rather than a like-for-like full-data fit. Permutation importance indicates predictive association in this test sample; it is not a causal explanation.

The existing dashboard also contains a rule-based kidney-risk flag using HbA1c, hypertension, age, BMI, and the diabetes prediction. That layer is a decision-support demonstration, not a kidney-disease diagnostic model, because the supplied dataset has no kidney outcome label or kidney-function measurements.

## 7 Conclusion

The project now provides a reproducible six-model benchmark with leakage-aware preprocessing, untouched-test evaluation, full metric exports, plots, and saved model artifacts. The recommended candidate is the model with the best balance of ROC-AUC, PR-AUC, recall, and F1 rather than the model with accuracy alone. Before any clinical interpretation, the selected model should undergo repeated cross-validation, threshold analysis, calibration assessment, subgroup checks, and external validation.

## References

Iftikhar, K., Javaid, N., Ahmed, I., & Alrajeh, N. (2025). A Novel Explainable Deep Learning Framework for Accurate Diabetes Mellitus Prediction. *Applied Sciences*, 15, 9162. Supplied project PDF: `applsci-15-09162__BASE paper.pdf`.
