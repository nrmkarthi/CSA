import matplotlib
import matplotlib.pyplot as plt
import os
FIG = os.environ.get('XAICSA_FIGS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper', 'figs'))
os.makedirs(FIG, exist_ok=True)
# validated categorical order (blue, orange, aqua, yellow) + neutral grey for baselines
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
GREY = '#8a8985'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'lines.linewidth': 1.4,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
})
