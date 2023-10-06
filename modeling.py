import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler


    
def modeling(df, hypo_list):
    '''
    Creates custom train/test splits and fits a model for each.
    Returns a plot with confusion matrix and AUC curve for each model, and evaluation metrics per model & overall metrics for all models.
    '''
   
    column = 0
    eval_metrics = dict()
    eval_metrics_list = []
    fig, axs = plt.subplots(2, len(hypo_list), figsize=(len(hypo_list)*4, 8))

    for i in hypo_list: 
        row = 0

        # train test split
        train = df[(df.day_night != i)]
        test = df[df.day_night == i]

        X_train = train.drop(['hypoglycemia', 'day_night', 'hypo_duration', 'sleep'], axis =1)
        X_test = test.drop(['hypoglycemia', 'day_night', 'hypo_duration', 'sleep'], axis=1)
        y_train = train[['hypoglycemia']]
        y_test = test[['hypoglycemia']]
    
        # define undersample strategy
        sampler = RandomUnderSampler(random_state=0, sampling_strategy=0.5)
    
        # undersampling train dataset
        X_train_under, y_train_under = sampler.fit_resample(X_train, y_train)
    
        # scaling train and test data
        robust_scaler = RobustScaler()

        X_train_ = robust_scaler.fit_transform(X_train_under)
        X_test_ = robust_scaler.transform(X_test)
    
        # baseline model
        #model = xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.7, reg_lambda=100)
        model = xgb.XGBClassifier(reg_alpha=10)

        # Fit the model to the training data
        model.fit(X_train_, y_train_under)
    
        # Make predictions on the testing data
        y_pred = model.predict(X_test_)
    
        # Evaluation metrics
        # Calculate precision
        precision = precision_score(y_test, y_pred)
    
        # Calculate recall
        recall = recall_score(y_test, y_pred)
    
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Display the confusion matrix as a heatmap
        sns.heatmap(cm, annot=True, cmap="Blues", fmt=".1f", linewidth=.5, ax=axs[row,column])
        axs[row,column].set_title('Day_ID: '+ str(i))
    
        # AUC 
        # Make predictions on the testing data
        y_scores = model.predict_proba(X_test_)[:, 1]

        # Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)

        # Calculate the area under the ROC curve (AUC)
        row +=1
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        axs[row, column].plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc)
        axs[row, column].plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
        axs[row, column].legend(loc="lower right")
    
        # populatin Evaluation metrics dict 
        eval_metrics['id'] = i
        eval_metrics['precision'] = precision
        eval_metrics['recall'] = recall
        eval_metrics['AUC'] = roc_auc
        eval_metrics_list.append(eval_metrics)
        eval_metrics = dict() #empty the dict for next set of values
    
        column +=1

    #plt.suptitle('XGBoost Model - tuned (awake)')
    plt.show()
    
    
    eval_metrics_df = pd.DataFrame(eval_metrics_list)
    eval_metrics_df.set_index('id')
    return('Evaluation metrics per night', eval_metrics_df, 'Average evaluation metrics for all nights:', eval_metrics_df.drop('id', axis=1).median(axis=0))
