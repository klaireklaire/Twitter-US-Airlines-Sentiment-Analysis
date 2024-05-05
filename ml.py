from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, roc_curve, f1_score, auc, accuracy_score
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

    print(classification_report(y_test, y_predicted, target_names=['negative', 'neutral', 'positive']))

    # Ignore 'neutral' in the confusion matrix
    non_neutral_indices = y_test != 'neutral'  # Assuming y_test is labeled with 'negative', 'neutral', 'positive'
    y_test_filtered = y_test[non_neutral_indices]
    y_predicted_filtered = y_predicted[non_neutral_indices]

    # Display confusion matrix
    labels = ['negative', 'positive']  # Labels after removing 'neutral'
    cm = confusion_matrix(y_test_filtered, y_predicted_filtered, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Calculate and print macro F1-score
    f1 = f1_score(y_test_filtered, y_predicted_filtered, average='macro')
    print("F1 score (macro, excluding neutral):", f1)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test_filtered, y_predicted_filtered)
    print("Accuracy (excluding neutral):", accuracy)
    #ConfusionMatrixDisplay.from_predictions(y_test, y_predicted)
    #plt.show()

    # print("F1 score:", str(f1_score(y_test, y_predicted, average='macro')))
    # print("Accuracy", str(accuracy_score(y_test, y_predicted)))

    return nb_best

def get_top_features(classifier, feature_names, top_n=20):
    """Retrieve top n features that contribute most to each class."""
    class_features = []
    for i, class_label in enumerate(classifier.classes_):
        top_indices = np.argsort(classifier.feature_log_prob_[i])[-top_n:]
        top_features = [(feature_names[j], classifier.feature_log_prob_[i][j]) for j in top_indices]
        class_features.append((class_label, top_features))
    
    class_features.sort(key=lambda x: x[0].lower() == 'positive', reverse=True)
    return class_features

def extract_words(class_features, class_label):
    [pos_tuples] = [x[1] for x in class_features if x[0] == class_label]
    words = []
    print(pos_tuples)
    for t in pos_tuples:
        words.append(t[0])
    print(words)
    return words 

def plot_top_words(class_features):
    fig, axes = plt.subplots(nrows=1, ncols=len(class_features), figsize=(12, 4))
    
    for ax, (class_label, features) in zip(axes, class_features):
        words, weights = zip(*features)
        indexes = np.arange(len(words))

        ax.barh(indexes, weights, color='darkcyan' if 'positive' in class_label else 'lightcoral')
        ax.set_yticks(indexes)
        # Set y-axis labels to bold and grey
        ax.set_yticklabels(words, color='grey')
        ax.set_xlabel('Weight of Keyword', fontweight='bold', color='grey')  # Set xlabel to bold and grey as well
        # Capitalize the first letter of the class label and set title to bold and grey
        ax.set_title(f'{class_label.capitalize()} Class', fontweight='bold', color='grey')
        ax.invert_yaxis()  # Highest weights at the top

    plt.tight_layout()
    plt.show()


