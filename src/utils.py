import matplotlib.cm as cm
import random

def detect_cluster_column(df):
    """
    :param df:
    :return: all segment cols
    """
    matches = [col for col in df.columns if "segment" in col.lower()]
    if len(matches) == 0:
        raise ValueError("No cluster column found (expected name containing 'Segment').")
    return matches[0]

def detect_components(df):
    """
    :param df:
    :return: all component cols
    """
    return [col for col in df.columns if col.startswith("Component")]

def generate_random_colors(N):
    """
    Generate N different Random colors
    :param N: number of colors to generate
    :return: array of colors
    """
    cmap = cm.get_cmap('tab20', N)
    palette = []

    for i in range(N):
        rand_color = cmap(random.randint(0, cmap.N-1))
        while rand_color in palette:
            rand_color = cmap(random.randint(0, cmap.N-1))
        palette.append(rand_color)

    return palette