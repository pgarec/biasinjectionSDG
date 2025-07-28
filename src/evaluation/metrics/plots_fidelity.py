import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

params = {'mathtext.default': 'regular'}
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'mathtext.default': 'regular',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
    'figure.figsize': (7, 5)  # Consistent size for single plot
})

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])

np.random.seed(0)

""" Neutral Grey	#bfbfbf	Light Grey	For unhighlighted bars
Elegant Red	#d62728	Deep Red	Perfect for highlight / emphasis
Soft Blue	#1f77b4	Professional Blue	Alternate highlight color
Gold / Orange	#ff7f0e	Warm Accent	For secondary highlights
Green	#2ca02c	Balanced Green	For positive effects
Purple	#9467bd	Soft Purple	For additional groups
Brown	#8c564b	Earthy Brown	Rare use, low priority highlight
Pink	#e377c2	Soft Pink	Optional for differentiating groups
Dark Grey	#7f7f7f	Medium Grey	For other muted elements"""


def get_score_plot(sdv_report):
    score_plot = (
        sdv_report.get_visualization(property_name="Column Shapes")
        .update_xaxes(tickfont_size=12, tickangle=-30)
        .update_yaxes(tickfont_size=12)
        .update_layout(
            legend=dict(
                orientation="h",
                xanchor="right",
                x=1,
                yanchor="bottom",
                y=1.02,  # Adjusted for better positioning
            ),
            title_text="Distribution of Column Shapes",  # More formal title
            title_x=0.5  # Center title
        )
    )
    return score_plot


def get_correlation_plot(sdv_report):
    corr_plot = (
        sdv_report.get_visualization(property_name="Column Pair Trends")
        .update_xaxes(tickfont_size=12, tickangle=-30)
        .update_yaxes(tickfont_size=12)
        .update_traces(colorscale="Viridis")
        .update_layout(
            legend=dict(
                orientation="h",
                xanchor="right",
                x=1,
                yanchor="bottom",
                y=1.02,  # Adjusted for better positioning
            ),
            title_text="Correlation Trends Between Columns",
            title_x=0.5  # Center title
        )
    )
    return corr_plot
