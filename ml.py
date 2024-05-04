from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, roc_curve, f1_score, auc, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def tune_train_evaluate_mnb_muticlass(X, y, X_train, X_test, y_train, y_test):
    nb = MultinomialNB()

    param_grid_nb = [
        {'alpha': [1.0e-10, 0.01, 0.1, 0.5, 1.0, 2.0, 10, 20, 50, 100],
         'fit_prior': [True, False]}
    ]

    clf_nb = GridSearchCV(nb, param_grid=param_grid_nb, cv=5, verbose=1, n_jobs=-1, scoring='f1_macro')
    best_clf_nb = clf_nb.fit(X, y)

    print(best_clf_nb.best_score_, best_clf_nb.best_estimator_)

    df_best = pd.DataFrame(best_clf_nb.cv_results_)
    print("Average cross-validation F1 score for all combinations: " + str(df_best.loc[:, 'mean_test_score'].mean()))

    nb_best = best_clf_nb.best_estimator_
    nb_best.fit(X_train, y_train)
    y_predicted = nb_best.predict(X_test)

    #ConfusionMatrixDisplay.from_predictions(y_test, y_predicted)
    #plt.show()

    print("F1 score:", str(f1_score(y_test, y_predicted, average='macro')))
    print("Accuracy", str(accuracy_score(y_test, y_predicted)))

    return nb_best

def get_top_features(classifier, feature_names, top_n=500):
    """Retrieve top n features that contribute most to each class."""
    # tuple
    res = []
    for i, class_label in enumerate(classifier.classes_):
        top_positive_features = np.argsort(classifier.feature_log_prob_[i])[-top_n:]
        #top_negative_features = np.argsort(classifier.feature_log_prob_[i])[:top_n]

        print(f"Top positive words for class {class_label}:")
        res.append([feature_names[j] for j in top_positive_features])
        #print([feature_names[j] for j in top_positive_features])
    return res
    

        #print(f"Top negative words for class {class_label}:")
        #print([feature_names[j] for j in top_negative_features])

