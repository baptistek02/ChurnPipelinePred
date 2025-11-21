import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np

def make_profiles(df, n_components=10, n_clusters=4):
    """
    Make clusters of a df, Kmeans with PCA reduction
    :param df: needs to be cleaned, pd.dummies()
    :param n_components: use cumulative_explained_variance()
    :param n_clusters: use determine_cluster_numbers()
    :return: df with components and segment col, silhouette score
    """
    # Normalize
    scaler = StandardScaler()
    standardized_df = scaler.fit_transform(df)

    # Feature Reduction
    pca = PCA(n_components)
    pca.fit(standardized_df)
    scores_pca = pca.transform(standardized_df)

    # Model
    kmeans = KMeans(n_clusters, random_state=42, n_init=10).fit(scores_pca)

    # Analysis
    df_segm_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_segm_pca_kmeans.columns.values[-n_components:] = [f'Component {i}' for i in range(n_components)]
    df_segm_pca_kmeans['Segment K_means PCA'] = kmeans.labels_

    # Silhouette Score
    sil_score = silhouette_score(scores_pca, kmeans.labels_)

    return df_segm_pca_kmeans, sil_score

def train_classifier(model, X_train, X_test, y_train, threshold=0.3):
    """
    Train a classifier on X_train, X_test, y_train, and threshold
    :param model: classifier
    :param X_train: needs to be normalized
    :param X_test: needs to be normalized
    :param y_train: needs to be caterogical
    :param threshold:
    :return: classifier, y_pred, y_proba
    """
    clf = model.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > threshold).astype(int)
    return clf, y_pred, y_proba

def stack_model(
        X_train_normalized,
        X_test_normalized,
        y_train,
        threashold=0.3
):
    """
    Stacking with Logistic Regression, RandomForestClassifier, GradientBoostingClassifier
    :param X_train_normalized: needs to be normalized
    :param X_test_normalized: needs to be normalized
    :param y_train: needs to be caterogical
    :param threashold:
    :return: classifier, y_pred, y_proba
    """

    base_learners = [
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=None,
                                      random_state=42, class_weight='balanced',
                                      max_features='sqrt')),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                          max_depth=3, random_state=0))
    ]

    meta_learner = LogisticRegression()

    clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        passthrough=False
    )

    clf.fit(X_train_normalized, y_train)

    # y_pred = clf.predict(X_test_normalized)
    y_proba = clf.predict_proba(X_test_normalized)[:, 1]
    y_pred = (y_proba > threashold).astype(int)  # seuil 0.3

    return clf, y_pred, y_proba

def get_metrics(
        clf,
        X_train_normalized,
        y_train,
        y_test,
        y_pred,
        y_proba,
):
    """
    Get metrics for a classifier model
    :param clf: classifier
    :param X_train_normalized: needs to be normalized
    :param y_train: needs to be categorical
    :param y_test: needs to be categorical
    :param y_pred:
    :param y_proba:
    :return: a dictionnary of metrics : accuracy, auc, confusion matrix, f1 score, cross validation score
    """
    return {
        "model_accuracy" : accuracy_score(y_test, y_pred),
        "model_auc" : roc_auc_score(y_test, y_proba),
        "model_confusion_matrix" : confusion_matrix(y_test, y_pred),
        "model_f1_score" : f1_score(y_test, y_pred, average=None),
        "model_cross_validation" : np.mean(cross_val_score(clf, X_train_normalized, y_train, cv=5))
    }




