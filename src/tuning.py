from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_gradient_boost_best_params(X_train_normalized, y_train):
    """
    :param X_train_normalized: features columns need to be normalized
    :param y_train: target must be categorical
    :return: best params and score for gradient boost model
    """
    param_grid_gb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }

    grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=0),
                           param_grid_gb, cv=5, scoring='f1')
    grid_gb.fit(X_train_normalized, y_train)

    return grid_gb.best_params_, grid_gb.best_score_

def get_random_forest_best_params(X_train_normalized, y_train):
    """
    :param X_train_normalized: features columns need to be normalized
    :param y_train: target must be categorical
    :return: best params and score for random forest
    """
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }

    grid_rf = GridSearchCV(RandomForestClassifier(random_state=0),
                           param_grid_rf, cv=5, scoring='f1')
    grid_rf.fit(X_train_normalized, y_train)

    return grid_rf.best_params_, grid_rf.best_score_

# Keep 80% of Cumuluative Explained Variance to know how many components you need to keep
def cumulative_explained_variance(df):
    """
    Display a plot of explained variance by components.
    :param df:
    :return:
    """
    scaler = StandardScaler()
    standardized_df = scaler.fit_transform(df)

    pca = PCA()
    pca.fit(standardized_df)

    plt.figure(figsize=(10,8))
    plt.plot(range(42), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.ylabel('Cumuluative Explained Variance')
    plt.xlabel('Number of Components')
    plt.title('Explained Variance by Components')
    plt.legend("Keep 80% of Cumuluative Explained Variance")
    plt.show()

# Elbow Method to choose the number of clusters for n components, for KMeans
def determine_cluster_numbers(df, n_components):
    """
    Display a plot that show for n cluster, the inertia
    :param df:
    :param n_components: use cumulative_explained_variance() before if needed
    :return:
    """
    scaler = StandardScaler()
    standardized_df = scaler.fit_transform(df)

    pca = PCA(n_components)
    pca.fit(standardized_df)
    scores_pca = pca.transform(standardized_df)

    wcss = []
    for i in range(1, 21):
        kmeans_pca =  KMeans(n_clusters=i, random_state=42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)

    plt.figure(figsize=(10,8))
    plt.plot(range(1,21), wcss, marker = 'o', linestyle = '--')
    plt.ylabel('WCSS')
    plt.title('Kmeans with PCA Clustering')
    plt.show()