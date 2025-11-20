from src.models import make_profiles, get_metrics, stack_model
from src.processing import load_and_clean, split_data
from src.generatePDF import generate_report
from src.utils import generate_random_colors
import pandas as pd

def main():
    df = load_and_clean("data/data.csv")
    df_segm_pca_kmeans, sil_score = make_profiles(df)
    n_clusters = df_segm_pca_kmeans['Segment K_means PCA'].nunique()
    cluster_colors = generate_random_colors(n_clusters)

    dft = df_segm_pca_kmeans.drop(columns=[f'Component {i}' for i in range(10)])
    dft = pd.get_dummies(dft, columns=['Segment K_means PCA'], drop_first=True, dtype=int)

    X_train_normalized, X_test_normalized, y_train, y_test = split_data(dft)
    model, y_pred, y_proba = stack_model(X_train_normalized, X_test_normalized, y_train)
    stack_metrics = get_metrics(model, X_train_normalized, y_train, y_test, y_pred, y_proba)

    generate_report(
        df=df, df_segm_pca_kmeans=df_segm_pca_kmeans,
        sil_score=sil_score,
        y_test=y_test,
        y_proba=y_proba,
        model=model,
        model_metrics=stack_metrics,
        palette=cluster_colors
    )

if __name__ == "__main__":
    main()