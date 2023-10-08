import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/Users/ankit/PycharmProjects/employeeAttrition/data_files/final data.csv').drop('Over18', axis=1)

#Extracting average work durations for each employee and number of leaves, i.e. , 0 duration days
df['avg_work_duration'] = df.iloc[:, 29:].mean(axis=1)
df['num_of_leaves'] = df.iloc[:, 29:].apply(lambda x: x.eq(0).sum(), axis=1)

df = df.drop(columns=list(df.columns)[29:290])

categorical_columns = ['Gender', 'BusinessTravel', 'Department',
                       'EducationField', 'MaritalStatus', 'JobRole']

cat_df = df[categorical_columns]

one_hot_encoded = pd.get_dummies(cat_df)

df = df.drop(columns=categorical_columns)

df1 = pd.concat([df, one_hot_encoded], axis=1)

df1['Attrition'] = df1['Attrition'].map({'Yes':1, 'No':0})

df1 = df1.fillna(0)

experiment_name = "AttritionPrediction"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id


def log_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save the confusion matrix as an artifact in MLflow
    plt.savefig(run_name + '_confusion_matrix.png')
    mlflow.log_artifact(run_name + '_confusion_matrix.png')
    plt.close()

with mlflow.start_run(run_name='Random Forest Classifier default', experiment_id=experiment_id):
    data = df1
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Attrition'], axis=1), data['Attrition'], test_size=0.20, random_state=42)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)

    # Get predicted labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric('Precision', precision)
    mlflow.log_metric('Recall', recall)
    mlflow.log_metric('F-1 score', f1)
    log_confusion_matrix(y_test, y_pred, 'Random Forest Classifier default')

with mlflow.start_run(run_name='Random Forest Classifier HP tuning', experiment_id=experiment_id):
    data = df1
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Attrition'], axis=1), data['Attrition'], test_size=0.20, random_state=42)
    # Define the parameter grid for randomized search

    param_grid = {
        'min_samples_leaf': np.arange(1, 20),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': np.arange(1, 20),
        'n_estimators': [50, 100, 200, 300]
    }

    clf = RandomForestClassifier(random_state=0)

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, cv=5, random_state=42, n_jobs=-1)

    # Fit the randomized search to the training data
    random_search.fit(X_train, y_train)

    # Get the best model from the randomized search
    best_clf = random_search.best_estimator_

    # Get predicted labels for the test set using the best model
    y_pred = best_clf.predict(X_test)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric('Precision', precision)
    mlflow.log_metric('Recall', recall)
    mlflow.log_metric('F-1 score', f1)
    log_confusion_matrix(y_test, y_pred, 'Random Forest Classifier HP tuning')

# XGBoost Classifier
with mlflow.start_run(run_name='XGBoost Classifier', experiment_id=experiment_id):
    data = df1
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Attrition'], axis=1), data['Attrition'],
                                                        test_size=0.20, random_state=42)

    # Define XGBoost parameters (customize as needed)
    params = {
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    clf = xgb.XGBClassifier(**params)

    # Fit the XGBoost model to the training data
    clf.fit(X_train, y_train)

    # Get predicted labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric('Precision', precision)
    mlflow.log_metric('Recall', recall)
    mlflow.log_metric('F-1 score', f1)
    log_confusion_matrix(y_test, y_pred, 'XGBoost Classifier')

with mlflow.start_run(run_name='XGBoost Classifier HP tuning', experiment_id=experiment_id):
    data = df1
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Attrition'], axis=1), data['Attrition'],
                                                        test_size=0.20, random_state=42)

    # Define the parameter grid for randomized search
    param_grid = {
        'max_depth': np.arange(1, 10),
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [50, 100, 200, 300],
        'min_child_weight': np.arange(1, 10),
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.2, 0.3],
        'reg_lambda': [0, 0.1, 0.2, 0.3]
    }

    clf = xgb.XGBClassifier()

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, cv=5, random_state=42, n_jobs=-1)

    # Fit the randomized search to the training data
    random_search.fit(X_train, y_train)

    # Get the best model from the randomized search
    best_clf = random_search.best_estimator_

    # Get predicted labels for the test set using the best model
    y_pred = best_clf.predict(X_test)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric('Precision', precision)
    mlflow.log_metric('Recall', recall)
    mlflow.log_metric('F-1 score', f1)
    log_confusion_matrix(y_test, y_pred, 'XGBoost Classifier HP tuning')

#Next steps - 1) xgboost tuning
#             2) deployment/docker packaging of the best model
#
# https://mlflow.org/docs/latest/quickstart_mlops.html

#mlflow.delete_experiment(experiment_id=experiment_id)
#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