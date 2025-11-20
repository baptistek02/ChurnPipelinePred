import matplotlib.cm as cm
import random

def generate_random_colors(N):
    cmap = cm.get_cmap('tab20', N)
    palette = []

    for i in range(N):
        rand_color = cmap(random.randint(0, cmap.N-1))
        while rand_color in palette:
            rand_color = cmap(random.randint(0, cmap.N-1))
        palette.append(rand_color)

    return palette