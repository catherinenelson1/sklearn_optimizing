__author__ = 'Cat'

# import main modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_training_testing_data(is_scaled, which_elements):
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler

    if which_elements == 'major':
        df = pd.read_csv('LP_major_element_recalc.csv', nrows=1260)

    if which_elements == 'major and xrf trace':
        df = pd.read_csv('LP_major_xrf_trace_recalc.csv', nrows=784)

    if which_elements == 'major and icp-ms':
        df = pd.read_csv('', nrows=0)

    X_train, X_rest, y_train, y_rest = train_test_split(df.iloc[1:, 1:], df.iloc[1:, 0], test_size=0.40,
                                                        random_state=42)
    X_crossval = X_rest[:len(X_rest) / 2]
    y_crossval = y_rest[:len(y_rest) / 2]
    X_test = X_rest[len(X_rest) / 2:]
    y_test = y_rest[len(y_rest) / 2:]

    if is_scaled == True:
        X_train = StandardScaler().fit_transform(X_train)
        X_crossval = StandardScaler().fit_transform(X_crossval)

    el_names = df.columns.values

    return X_train, X_crossval, X_test, y_train, y_crossval, y_test, el_names

X_train, X_crossval, X_test, y_train, y_crossval, y_test, el_names = make_training_testing_data(False, 'major and xrf trace')


def classify_all(X_train, y_train, X_crossval, y_crossval):
    # import all the classifiers
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn import linear_model

    # make a dictionary of classifier names and sklearn calls
    clf_dict = {
        "Nearest Neighbors": {'call': KNeighborsClassifier(algorithm='auto'), 'params': {'n_neighbors': [1, 3, 5]}},
        "SVM": {'call': SVC(), 'params': {'kernel': ('linear', 'rbf'), 'C': [100, 1000],
                                          'gamma': [0.0001, 0.001, 0.01, 0.1]}},
        "Decision Tree": {'call': DecisionTreeClassifier(), 'params': {'max_depth': [5, 3, None]}},
        "Random Forest": {'call': RandomForestClassifier(n_estimators=100),
                          'params': {"max_depth": [5, 3, None], "max_features": [1, 10],
                                     "min_samples_split": [1, 10], "min_samples_leaf": [1, 10],
                                     "criterion": ["gini", "entropy"]}},
        "AdaBoost": {'call': AdaBoostClassifier(), 'params': {}},
        "Naive Bayes": {'call': GaussianNB(), 'params': {}},
        "Logistic Regression": {'call': linear_model.LogisticRegression(), 'params': {'C': [0.1, 1, 10]}}
    }

    results = []
    from sklearn.grid_search import GridSearchCV

    for clf in clf_dict:
        model = GridSearchCV(clf_dict[clf]['call'], clf_dict[clf]['params'])
        model.fit(X_train, y_train)
        score = model.score(X_crossval, y_crossval)
        results.append([clf, score, model.best_params_])
    return results

print classify_all(X_train, y_train, X_crossval, y_crossval)
# ['Random Forest', 0.95634920634920639, {'max_features': 10, 'min_samples_split': 1, 'criterion': 'entropy',
# 'max_depth': None, 'min_samples_leaf': 1}],


def view_forest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_features=10, min_samples_split=1, criterion='entropy',
                                 max_depth=None, min_samples_leaf=1)
    clf.fit(X_train, y_train)
    imps = clf.feature_importances_

    ind = np.arange(len(imps))
    width = 0.5
    fig, ax = plt.subplots()
    plt.bar(ind, imps, width)
    plt.xticks(ind + width, el_names[1:])
    plt.show()


view_forest()

# results using trace element data, not scaled: [['SVM', 0.92993630573248409,
# {'kernel': 'rbf', 'C': 100, 'gamma': 0.0001}],
# ['Decision Tree', 0.87898089171974525, {'max_depth': 5}], ['Naive Bayes', 0.91082802547770703, {}],
# ['AdaBoost', 0.95541401273885351, {}], ['Logistic Regression', 0.95541401273885351, {'C': 0.1}],
# ['Random Forest', 0.95541401273885351, {'max_features': 10, 'min_samples_split': 10,
# 'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1}],
# ['Nearest Neighbors', 0.90445859872611467, {'n_neighbors': 3}]]
