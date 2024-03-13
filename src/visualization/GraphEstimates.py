from Metrics import PlottingData
import matplotlib.pyplot as plt
import numpy as np

def plot_eigenvalues(ax, initialEigenvalues, finalEigenvalues):
    barWidth = 0.1


    # Set position of bar on X axis
    br1 = np.arange(len(initialEigenvalues))
    br2 = [x + barWidth for x in br1]

    ax.bar(br1, initialEigenvalues, color='r', width=barWidth,
            edgecolor='grey', label='IT')
    ax.bar(br2, finalEigenvalues, color='g', width=barWidth,
            edgecolor='grey', label='ECE')