# Assignment 4: Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
A healthcare organization wants to develop a machine learning model to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic measurements. This project develops a K-Nearest Neighbors (KNN) classification model to perform this classification.

## Dataset
**Breast Cancer Wisconsin Diagnostic Dataset**
Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

> Note: The raw dataset is not uploaded to this repository (per assignment instructions). The notebook loads the identical UCI Breast Cancer Wisconsin Diagnostic dataset via `sklearn.datasets.load_breast_cancer()` for reproducibility. If you'd rather use the Kaggle CSV directly, download `data.csv` from the link above, place it in this folder, and uncomment the `pd.read_csv("data.csv")` line at the top of the notebook.

## Libraries Used
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, identified 30 numerical diagnostic features and the target variable (`diagnosis`), and reviewed dataset info and summary statistics.
2. **Data Preprocessing** — Checked for missing values (none found), dropped unnecessary ID columns where present, label-encoded the target (`B` = 0, `M` = 1), standardized all feature values using `StandardScaler`, and split the data 80/20 into training and test sets (stratified by class).
3. **Model Development** — Trained a `KNeighborsClassifier` with **K = 5** on the standardized training data and generated predictions on the test set.
4. **Model Evaluation** — Evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix. Also plotted accuracy across K values from 1–20 to visualize model stability.

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.956 |
| Precision | 0.974 |
| Recall | 0.905 |
| F1-Score | 0.938 |

**Confusion Matrix:**

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 71 | 1 |
| **Actual Malignant** | 4 | 38 |

### Observations
1. The KNN model with K = 5 achieves high accuracy (95.6%) on the test set, showing strong separability between malignant and benign classes based on diagnostic measurements.
2. Very few misclassifications occurred; the 4 false negatives (malignant predicted as benign) are the most clinically important errors to minimize, even though overall accuracy is high.
3. Accuracy remains fairly stable across a range of K values (see `k_vs_accuracy.png`), suggesting the classes are well-separated after standardization.

## Conclusion
The K-Nearest Neighbors model built on the Breast Cancer Wisconsin Diagnostic dataset achieved strong classification performance, with high accuracy, precision, recall, and F1-score on the held-out test set. The model successfully distinguishes malignant from benign tumors using diagnostic features such as radius, texture, perimeter, and concavity of cell nuclei.

Feature scaling proved essential for this task. Since KNN classifies points based on distance (typically Euclidean), features measured on larger numeric scales would otherwise dominate the distance calculation and bias predictions, even if they aren't more informative. Standardizing all features to a common scale ensured each contributed fairly to the neighbor search.

A key limitation of KNN is its computational cost at prediction time: since it stores the entire training set and computes distances to all training points for every new prediction, it scales poorly to very large datasets and high-dimensional feature spaces. It is also sensitive to irrelevant or noisy features and to the choice of K.

## Repository Structure
```
.
├── Assignment-4.ipynb      # Full notebook with code, outputs, and analysis
├── README.md                # This file
├── confusion_matrix.png     # Confusion matrix visualization
└── k_vs_accuracy.png        # Accuracy vs K value plot
```
