# Main Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import  SVC,LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold,train_test_split,cross_val_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


########## train_using_logistic_regression ##########

def create_dataframe(df_data, df_columns, show_after_creation = False):
    dataframe = pd.DataFrame(data= df_data, columns= df_columns)
    if(show_after_creation == True):
        print(dataframe)
    return dataframe

########## show_accuracy_results ##########
def show_accuracy_results(cm, accuracy, precision, recall, f1):
    # Show Accuracy results
    print('Confusion matrix:\n', cm)
    print('\nAccuracy : %.3f' %accuracy)
    print('Precision : %.3f' %precision)
    print('Recall : %.3f' %recall)
    print('F1-score : %.3f' %f1)

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm, index = ['setosa','versicolor','virginica'], columns = ['setosa','versicolor','virginica'])
    
    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Accuracy:{0:.3f}'.format(accuracy))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

########## train_using_logistic_regression ##########
def train_model_using_logistic_regression(x_train, x_test, y_train, y_test, show_accuracy_results = False):
    
    reg=LogisticRegression()
    
    # Training/Fitting the Model
    predModel = reg.fit(x_train,y_train)
    
    # Making Predictions
    y_pred = predModel.predict(x_test)

    # Calculate Performance
    cm = confusion_matrix(y_test, y_pred) # Confusion Matrix
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    f1 = f1_score(y_test,y_pred,average='micro')

    if(show_accuracy_results):
        show_accuracy_results(accuracy)

    return cm, accuracy, precision, recall, f1


########## train_using_support_vector_classification ##########
def train_model_using_support_vector_classification(x_train, x_test, y_train, y_test, show_accuracy_results = False):

    # Training/Fitting the Model
    clf = SVC(kernel = 'linear').fit(x_train, y_train)
    
    # Making Predictions
    y_pred = clf.predict(x_test)

    # Calculate Performance
    cm = confusion_matrix(y_test, y_pred) # Confusion Matrix
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    f1 = f1_score(y_test,y_pred,average='micro')

    if(show_accuracy_results):
        show_accuracy_results(accuracy)

    return cm, accuracy, precision, recall, f1

########## train_using_random_forest ##########
def train_using_random_forest(x_train, x_test, y_train, y_test, show_accuracy_results = False):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    
    # Making Predictions
    y_pred = random_forest.predict(x_test)

    # Calculate Performance
    cm = confusion_matrix(y_test, y_pred) # Confusion Matrix
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    f1 = f1_score(y_test,y_pred,average='micro')
    
    return cm, accuracy, precision, recall, f1

########## scale_features_using_min_max_scaler ##########
def scale_features_using_min_max_scaler(data):
    # data is Features
    min_max_scaler = MinMaxScaler()
    min_max_data_scaled = min_max_scaler.fit_transform(data)
    return min_max_data_scaled

########## scale_features_using_standard_scaler ##########
def scale_features_using_standard_scaler(data):
    # data is Features
    standard_scaler = StandardScaler()
    standard_data_scaled = standard_scaler.fit_transform(data)
    return standard_data_scaled

########## compare_logistic_regression_vs_vector_classification_accuracy ##########
def compare_logistic_regression_vs_vector_classification_accuracy(accuracyLR, accuracySVC):
    results = pd.DataFrame({'Model': [ 'Logistic Regression', ' Support Vector Classification'], "Accuracy_score": [accuracyLR, accuracySVC]})
    result_df = results.sort_values(by='Accuracy_score', ascending=False)
    result_df = result_df.reset_index(drop=True)
    result_df.head(50)
    print(result_df)

