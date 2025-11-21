from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from src.visualization import make_cluster_plot, make_spider_plot, make_bar_churn_plot, make_roc_curve

def generate_report(
        df,
        df_segm_pca_kmeans,
        sil_score,
        y_test,
        y_proba,
        model,
        model_metrics,
        palette=None,
        title="Churn report",
        objective="Objective",
        ):
    """
    Generate a PDF Churn Report
    :param df: cleaned df
    :param df_segm_pca_kmeans: df with components cols and a segment col
    :param sil_score: silhouette score
    :param y_test:
    :param y_proba:
    :param model: classifier model
    :param model_metrics: metrics of the classifier model
    :param palette: color palette if None then random colors will be used
    :param title: title of the report
    :param objective: Objective of the report
    :return:
    """
    doc = SimpleDocTemplate("./output/report.pdf", pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(objective, styles['BodyText']))
    elements.append(Spacer(1, 12))

    # Cluster Plot
    make_cluster_plot(df_segm_pca_kmeans, palette=palette)

    cluster_plot = Image("./assets/cluster_plot.png", width=400, height=300)
    elements.append(cluster_plot)
    elements.append(Paragraph(f"Silhouette Score: {sil_score}", styles['BodyText']))
    elements.append(Spacer(1, 12))

    # Bar Plot Churn Mean by Profile
    make_bar_churn_plot(df_segm_pca_kmeans, palette=palette)

    bar_churn_plot = Image("./assets/bar_churn_plot.png", width=400, height=300)
    elements.append(bar_churn_plot)

    # Spider Plot
    make_spider_plot(df, df_segm_pca_kmeans, palette=palette)

    spider_plot = Image("./assets/spider_plot.png", width=400, height=300)
    elements.append(spider_plot)

    # ROC Curve Classifier Model
    make_roc_curve(y_test, y_proba, model, output_path="./assets/roc_curve.png")

    roc_curve = Image("./assets/roc_curve.png", width=400, height=300)
    elements.append(roc_curve)
    for key, value in model_metrics.items():
        elements.append(Paragraph(f"{key}: {value}", styles['BodyText']))
    elements.append(Spacer(1, 12))

    doc.build(elements)
