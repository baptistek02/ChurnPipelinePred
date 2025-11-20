import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, RocCurveDisplay
from src.utils import generate_random_colors

def detect_components(df):
    return [col for col in df.columns if col.startswith("Component")]


def detect_cluster_column(df):
    matches = [col for col in df.columns if "segment" in col.lower()]
    if len(matches) == 0:
        raise ValueError("No cluster column found (expected name containing 'Segment').")
    return matches[0]


def make_cluster_plot(df, output_path="./assets/cluster_plot.png", palette=None):
    components = detect_components(df)
    if len(components) < 2:
        raise ValueError("At least 2 PCA components are required.")

    cluster_col = detect_cluster_column(df)
    n_clusters = df[cluster_col].nunique()

    # palette
    if palette is None:
        palette = generate_random_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(
        x=df[components[0]],
        y=df[components[1]],
        hue=df[cluster_col],
        palette=palette,
        ax=ax
    )

    ax.set_title("Clusters by PCA Components")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_spider_plot(df_original, df_segm, target="Churn",
                     output_path="./assets/spider_plot.png",
                     palette=None):

    components = detect_components(df_segm)
    cluster_col = detect_cluster_column(df_segm)
    n_components = len(components)

    # cluster means on original variables
    cluster_means = df_segm.drop(columns=components).groupby(cluster_col).mean()

    # select correlated factors
    corr = cluster_means.corr()[target]
    factors = corr[(abs(corr) >= 0.7)].index

    # normalize 0-100
    dft = pd.DataFrame()
    for factor in factors:
        min_val = df_original[factor].min()
        max_val = df_original[factor].max()
        dft[factor] = cluster_means[factor].apply(lambda x:
            100 * (x - min_val) / (max_val - min_val)
        )

    labels = dft.columns
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    if palette is None:
        palette = generate_random_colors(dft.shape[0])

    for idx, color in enumerate(palette):
        values = dft.iloc[idx].tolist() + [dft.iloc[idx, 0]]
        ax.plot(angles, values, color=color, linewidth=1)
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_title("Profile Radar Chart", y=1.08)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.legend(dft.index, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_bar_churn_plot(df_segm, target="Churn",
                        output_path="./assets/bar_churn_plot.png",
                        palette=None):

    components = detect_components(df_segm)
    cluster_col = detect_cluster_column(df_segm)

    cluster_means = df_segm.drop(columns=components).groupby(cluster_col).mean()

    labels = cluster_means.index
    values = cluster_means[target].values * 100

    if palette is None:
        palette = generate_random_colors(len(labels))

    fig, ax = plt.subplots()

    ax.bar(labels, values, color=palette)
    ax.set_ylabel("Churn %")
    ax.set_title("Churn Risk by Cluster")
    ax.set_ylim(0, 100)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def make_roc_curve(
        y_test,
        y_proba,
        model,
        output_path="./assets/roc_curve.png",
):
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=model.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    fig, ax = plt.subplots(figsize=(10, 8))
    roc_display.plot(ax=ax)
    ax.set_title("ROC Curve")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

