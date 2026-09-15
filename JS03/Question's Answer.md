1. Differentiate variables that can be used and cannot be used

The id variable cannot be used as a predictive feature because it is only an identifier for each patient and does not represent a medical characteristic. The Unnamed: 32 column also cannot be used because it is an empty column containing only missing values.

The diagnosis variable is used as the target variable, while the remaining 30 numerical variables are used as predictive features.

2. Encode the diagnosis variable

The diagnosis variable contains two classes:

M = Malignant → 1
B = Benign → 0

This encoding converts the categorical target into numerical values that can be processed by Logistic Regression.

3. Standardize all numerical columns

All 30 numerical features were standardized using StandardScaler.

Standardization is used to make the features have a comparable scale, with a mean close to 0 and a standard deviation close to 1.

4. Perform feature selection

Feature selection was performed using SelectKBest with the f_classif scoring function.

Several values of k were tested:

Number of Features	Accuracy
5	                96.49%
10                 	95.61%
15	                97.37%
20	                98.25%
25	                97.37%
30	                96.49%

The best result was obtained using 20 features.

5. Test using Logistic Regression

The selected features were tested using Logistic Regression.
The best model achieved an accuracy of:

98.25%

The classification report also showed strong performance:

- Class 0: precision 0.97, recall 1.00, F1-score 0.99
- Class 1: precision 1.00, recall 0.95, F1-score 0.98
- Overall accuracy: 0.98

This indicates that Logistic Regression performs very well on this dataset after feature selection and standardization.

6. Model Pipeline

A pipeline was used to combine the preprocessing and machine learning steps:

StandardScaler
      ↓
SelectKBest
      ↓
Logistic Regression

Using a pipeline helps ensure that the preprocessing and feature selection are performed consistently as part of the model training process.

7. How many optimal features and which features?

Based on the experiment, the optimal number of features is 20, because it produced the highest accuracy of 98.25%.

The selected features are:

radius_mean
texture_mean
perimeter_mean
area_mean
compactness_mean
concavity_mean
concave points_mean
radius_se
perimeter_se
area_se
concave points_se
radius_worst
texture_worst
perimeter_worst
area_worst
smoothness_worst
compactness_worst
concavity_worst
concave points_worst
symmetry_worst

Conclusion

After removing the id and Unnamed: 32 variables, 30 numerical features were available for analysis. The diagnosis variable was encoded into binary values, with Malignant as 1 and Benign as 0. The numerical features were standardized using StandardScaler, followed by feature selection using SelectKBest. Several numbers of selected features were tested using Logistic Regression. The best performance was obtained using 20 features, achieving an accuracy of 98.25%. Therefore, 20 features are considered the optimal number of features based on the experiments conducted.