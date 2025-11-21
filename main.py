from src.models import make_profiles, get_metrics, stack_model, predict
from src.processing import load_and_clean, split_data, clean_na_duplicates, encode, clear_features
from src.generatePDF import generate_report
from src.utils import generate_random_colors
from sklearn.datasets import make_classification
import pandas as pd

def main():

    # Train model
    df = load_and_clean("data/data.csv")

    # Churn profiles
    df_segm_pca_kmeans, sil_score = make_profiles(df)
    n_clusters = df_segm_pca_kmeans['Segment K_means PCA'].nunique()
    cluster_colors = generate_random_colors(n_clusters)

    # Churn classifier model
    X_train_normalized, X_test_normalized, y_train, y_test = split_data(df)
    model, y_pred, y_proba = stack_model(X_train_normalized, X_test_normalized, y_train)
    stack_metrics = get_metrics(model, X_train_normalized, y_train, y_test, y_pred, y_proba)

    # Prediction
    df_predict = pd.read_csv("data/test.csv", sep=",")
    df_predict = clean_na_duplicates(df_predict)
    dft = encode(df_predict)
    dft = clear_features(dft)
    y_proba, y_pred = predict(model, dft)
    df_predict["Churn"] = y_pred
    df_predict["Probability"] = y_proba

    # PDF
    generate_report(
        df=df, df_segm_pca_kmeans=df_segm_pca_kmeans, df_predict=df_predict,
        sil_score=sil_score,
        y_test=y_test,
        y_proba=y_proba,
        model=model,
        model_metrics=stack_metrics,
        palette=cluster_colors
    )

if __name__ == "__main__":
    main()