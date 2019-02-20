from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns; sns.set()
from scipy import stats
import numpy.ma as ma
import time

"""
Load data and organize/subset for correlations between 16 song variables and 
latitude, longitude, and year 
"""
data_path = ("C:/Users/abiga/Box "
             "Sync/Abigail_Nicole/ChippiesTimeOfDay"
             "/FinalChippiesDataReExportedAs44100Hz_LogTransformed"
             "_forTOD.csv")
data_for_TOD = pd.DataFrame.from_csv(data_path, header=0, index_col=0)
print(data_for_TOD.shape)
data_for_TOD = data_for_TOD[data_for_TOD['RecordingTime'] < 12*3600]

print(data_for_TOD.columns)
print(data_for_TOD.RecordingTime)

"""
Continuous Stats Test on 16 chosen song variables (pearsons and spearmans, 
only use spearmans for the paper) 
"""

def corrfunc(x, y, **kws):
    rho, p_rho = stats.spearmanr(x, y, nan_policy='omit')
    ax = plt.gca()
    if p_rho < 0.05:
        weight = 'bold'
    else:
        weight = 'normal'

    try:
        ax.annotate("rho = {:.2f}, p = {:.2e}".format(rho, p_rho),
                    xy=(.1, 1), xycoords=ax.transAxes, fontsize=8, fontweight=weight)
    except ValueError:
        p_rho = float(ma.getdata(p_rho))
        ax.annotate("rho = {:.2f}, p = {:.2e}".format(rho, p_rho),
                    xy=(.1, 1), xycoords=ax.transAxes, fontsize=8, fontweight=weight)

    # if x.name == 'RecordingTime':
    #     labels = [item.get_text() for item in ax.get_xticklabels()]
    #     print(labels)
    #     print([time.strftime('%H:%M', time.gmtime(float(label)*100000)) for label in labels])
    #     ax.set_xticklabels([time.strftime('%H:%M', time.gmtime(float(label)*100000)) for label in labels])

# plotting correlations
pdf = PdfPages("C:/Users/abiga\Box "
               "Sync\Abigail_Nicole\ChippiesTimeOfDay"
               "\TODCorrelations"
               "/SongVar_corr_wReg_beforeNoon.pdf")
sns.set(style='white',
        rc={"font.style": "normal",
            'lines.markersize': 2,
            'axes.labelsize': 5,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8})

g = sns.pairplot(data=data_for_TOD,
                 x_vars=data_for_TOD.columns[3],
                 y_vars=data_for_TOD.columns[4:],
                 kind='reg')

g.map(corrfunc)
pdf.savefig(transparent=True)
pdf.close()
plt.show()