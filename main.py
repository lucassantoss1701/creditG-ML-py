from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


import pandas as pd

columns = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 
               'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans', 
               'housing', 'job', 'own_telephone', 'foreign_worker']

def pre_processing(creditG_data_set):

    creditG_data_set.to_csv('creditG_dataSet.csv', index=False)
    creditG_dataset_with_one_hot = pd.get_dummies(creditG_data_set, columns=columns)
    class_data = creditG_dataset_with_one_hot.drop(['class'], axis=1)
    labelEncoder = LabelEncoder()
    Y = creditG_dataset_with_one_hot['class'].values
    Y = labelEncoder.fit_transform(Y)
    return class_data, Y

def train_decision_tree(X_train, y_train):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    return decision_tree

def train_random_forest(X_train, y_train):
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    return random_forest

def train_logistic_regression(X_train, y_train):
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(X_train, y_train)
    return logistic_regression

def print_analysis(y_test, y_pred_dt, y_pred_rf, y_pred_lr):
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print("Decision Tree Accuracy:", accuracy_dt)
    print("Random Forest Accuracy:", accuracy_rf)
    print("Logistic Regression Accuracy:", accuracy_lr)

def visualize_decision_tree(decision_tree, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree, feature_names=feature_names, filled=True, rounded=True, class_names=["Bad", "Good"])
    plt.show()

def visualize_random_forest(random_forest, feature_names):
    fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (25,5), dpi=800)
    for index in range(0, 5):
        plot_tree(random_forest.estimators_[index], feature_names=feature_names, filled=True, ax = axes[index], class_names=["Bad", "Good"])
        axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
    fig.savefig('rf_5trees.png')
    fig.show()

def visualize_logistic_regression(logistic_regression, feature_names):
    coef = logistic_regression.coef_[0]
    feature_importance = pd.Series(coef, index=feature_names)
    feature_importance.plot(kind='barh', figsize=(10, 6))
    plt.title("Logistic Regression Feature Importance")
    plt.xlabel("Coefficient Magnitude")
    plt.ylabel("Feature")
    plt.show()

def main():
    creditG_data = fetch_openml(name='credit-g', version=1)
    creditG_dataSet = pd.DataFrame(creditG_data.data, columns=creditG_data.feature_names)
    creditG_dataSet['class'] = creditG_data.target
    
    class_data, Y = pre_processing(creditG_dataSet)
    
    X_train, X_test, y_train, y_test = train_test_split(class_data, Y, test_size=0.2, random_state=42)
    
    decision_tree = train_decision_tree(X_train, y_train)
    random_forest = train_random_forest(X_train, y_train)
    logistic_regression = train_logistic_regression(X_train, y_train)
    
    y_pred_dt = decision_tree.predict(X_test)
    y_pred_rf = random_forest.predict(X_test)
    y_pred_lr = logistic_regression.predict(X_test)
    
    print_analysis(y_test, y_pred_dt, y_pred_rf, y_pred_lr)

    new_customers = {
        'checking_status': ['<0', '0<=X<200', 'no checking', 'no checking', '<0'],
        'credit_history': ['critical/other existing credit', 'existing paid', 'delayed previously', 'existing paid', 'all paid'],
        'purpose': ['radio/tv', 'education', 'furniture/equipment', 'radio/tv', 'education'],
        'savings_status': ['<100', '500<=X<1000', 'no known savings', '<100', '<100'],
        'employment': ['<1', '1<=X<4', '4<=X<7', 'unemployed', '<1'],
        'personal_status': ['single male', 'female div/dep/mar', 'male single', 'male single', 'female div/dep/mar'],
        'other_parties': ['none', 'none', 'none', 'none', 'none'],
        'property_magnitude': ['real estate', 'real estate', 'real estate', 'life insurance', 'real estate'],
        'other_payment_plans': ['none', 'none', 'none', 'none', 'none'],
        'housing': ['own', 'own', 'own', 'own', 'own'],
        'job': ['skilled', 'unskilled resident', 'skilled', 'skilled', 'skilled'],
        'own_telephone': ['yes', 'yes', 'yes', 'yes', 'yes'],
        'foreign_worker': ['yes', 'yes', 'yes', 'yes', 'yes']
    }

    df_new_customers = pd.DataFrame(new_customers)

    df_new_customers_with_one_hot = pd.get_dummies(df_new_customers, columns=columns)

    missing_cols = set(class_data.columns) - set(df_new_customers_with_one_hot.columns)
    for col in missing_cols:
        df_new_customers_with_one_hot[col] = 0

    #reordering
    df_new_customers_with_one_hot = df_new_customers_with_one_hot[class_data.columns] 

    y_pred_novos_dt = decision_tree.predict(df_new_customers_with_one_hot)
    y_pred_novos_rf = random_forest.predict(df_new_customers_with_one_hot)
    y_pred_novos_lr = logistic_regression.predict(df_new_customers_with_one_hot)
    
    for i in range(len(new_customers['checking_status'])):
        print()
        print(f"Customer {i+1}:")
        print("Decision Tree:", "Good" if y_pred_novos_dt[i] == 1 else "Bad")
        print("Random Forest:", "Good" if y_pred_novos_rf[i] == 1 else "Bad")
        print("Logistic Regression:", "Good" if y_pred_novos_lr[i] == 1 else "Bad")
        print()

    visualize_decision_tree(decision_tree, class_data.columns)
    visualize_logistic_regression(logistic_regression, class_data.columns)
    visualize_random_forest(random_forest, class_data.columns)


main()
