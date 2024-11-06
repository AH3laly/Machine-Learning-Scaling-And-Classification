########## Scale data using StandardScaler ##########
import functions as fns
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
iris = load_iris()

# scale_features_using_standard_scaler
data_not_scaled = iris.data

# Display DataFrame
df_data = fns.create_dataframe(data_not_scaled, iris.feature_names)
df_data.head(20)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_not_scaled, iris.target, test_size=0.3)

# train_model_using_logistic_regression
cm, accuracyLR, precision, recall, f1 = fns.train_model_using_logistic_regression(X_train, X_test, y_train, y_test, True)

# train_model_using_support_vector_classification
cm, accuracySVC, precision, recall, f1 = fns.train_model_using_support_vector_classification(X_train, X_test, y_train, y_test, True)

# compare_logistic_regression_vs_vector_classification_accuracy
fns.compare_logistic_regression_vs_vector_classification_accuracy(accuracyLR, accuracySVC);

