# Main Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


########## train_using_logistic_regression ##########

def create_dataframe(df_data, df_columns, show_after_creation = False):
    dataframe = pd.DataFrame(data= df_data, columns= df_columns)
    if(show_after_creation == True):
        print(dataframe)
    return dataframe

########## train_using_logistic_regression ##########
def train_model_using_logistic_regression(x_train, x_test, y_train, y_test, show_accuracy_results = False):
    
    reg=LogisticRegression()
    
    # Training/Fitting the Model
    predModel = reg.fit(x_train,y_train)
    
    # Making Predictions
    y_predLR = predModel.predict(x_test)

    # Calculate Performance
    cm = confusion_matrix(y_test, y_predLR) # Confusion Matrix
    accuracyLR = accuracy_score(y_test,y_predLR)
    precision = precision_score(y_test, y_predLR,average='micro')
    recall = recall_score(y_test, y_predLR,average='micro')
    f1 = f1_score(y_test,y_predLR,average='micro')

    if(show_accuracy_results):
        # Show Predictios Results
        #print("Accuracy:", reg.score(x_test,y_test))
        
        # Show Accuracy results
        # print('Confusion matrix for Logistic Regression:\n', cm)
        print('\naccuracy_LR : %.3f' %accuracyLR)
        print('precision_LR : %.3f' %precision)
        print('recall_LR : %.3f' %recall)
        print('f1-score_LR : %.3f' %f1)

        # Transform to df for easier plotting
        cm_df = pd.DataFrame(cm, index = ['setosa','versicolor','virginica'], columns = ['setosa','versicolor','virginica'])
        
        plt.figure(figsize=(5.5,4))
        sns.heatmap(cm_df, annot=True)
        plt.title('Logistic Regression Accuracy:{0:.3f}'.format(accuracyLR))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    return cm, accuracyLR, precision, recall, f1


########## train_using_support_vector_classification ##########
def train_model_using_support_vector_classification(x_train, x_test, y_train, y_test, show_accuracy_results = False):

    # Training/Fitting the Model
    clf = SVC(kernel = 'linear').fit(x_train, y_train)
    
    # Making Predictions
    y_predSVC = clf.predict(x_test)

    # Calculate Performance
    cm = confusion_matrix(y_test, y_predSVC) # Confusion Matrix
    accuracySVC = accuracy_score(y_test,y_predSVC)
    precision = precision_score(y_test, y_predSVC,average='micro')
    recall = recall_score(y_test, y_predSVC,average='micro')
    f1 = f1_score(y_test,y_predSVC,average='micro')

    if(show_accuracy_results):
        # Show Predictios Results
        #print("Accuracy:", clf.score(x_test,y_test))
        
        # Show Accuracy results
        # print('Confusion matrix for SVC\n', cm)
        print('accuracy_SVC : %.3f' %accuracySVC)
        print('precision_SVC : %.3f' %precision)
        print('recall_SVC : %.3f' %recall)
        print('f1-score_SVC : %.3f' %f1)
    
        # Transform to df for easier plotting
        cm_df = pd.DataFrame(cm, index = ['setosa','versicolor','virginica'], columns = ['setosa','versicolor','virginica'])
        plt.figure(figsize=(5.5,4))
        sns.heatmap(cm_df, annot=True)
        plt.title('SVM Linear Kernel Accuracy:{0:.3f}'.format(accuracySVC))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    return cm, accuracySVC, precision, recall, f1

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

