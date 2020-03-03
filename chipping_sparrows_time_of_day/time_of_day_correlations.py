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
Load data and organize/subset wilcoxon rank sums test 
"""
# load in song data
data_path = "C:/Users/abiga/Box " \
            "Sync/Abigail_Nicole/ChippiesTimeOfDay" \
            "/FinalChippiesDataReExportedAs44100Hz_LogTransformed_forTOD.csv"
log_song_data = pd.DataFrame.from_csv(data_path, header=0, index_col=None)

col_to_skip = ['Latitude', 'Longitude', 'RecordingDay',
               'RecordingMonth', 'RecordingYear', 'RecordingTime',
               'RecordingTimeSeconds']
data_subset = log_song_data.drop(col_to_skip, axis=1)

# load in time data --> before or after sunrise, twilights, and noon (only going to use sunrise and noon)
data_path = "C:/Users/abiga/Box " \
            "Sync/Abigail_Nicole/ChippiesTimeOfDay" \
            "/FinalChippiesDataReExportedAs44100Hz_LogTransformed" \
            "_forTOD_SunriseTwilightNoon_JFORevisions.csv"
time_data = pd.DataFrame.from_csv(data_path, header=0, index_col=None)
# must remove duplicates -- have more than one bird from same recording -- duplicate catalog number and time data
time_data = time_data.drop_duplicates()

# combine tables using catalog no
combined_df = pd.merge(data_subset, time_data, on='CatalogNo')

# only keep ones with time data
combined_df = combined_df.drop(combined_df[combined_df.Sunrise ==
                                           '--'].index).copy().reset_index(
    drop=True)
combined_df['TimeDiffSec'] = combined_df['TimeDiffSec'].astype(float)

print(combined_df.columns)
print(combined_df.shape)
print(combined_df.groupby('Sunrise').count()['CatalogNo'])
song_variables = combined_df.columns[1:5]
print(song_variables)
print(min(combined_df.TimeDiffSec), max(combined_df.TimeDiffSec))
print(combined_df.columns[-1])


"""
Continuous Stats Test
"""

def corrfunc(x, y, **kws):
    rho, p_rho = stats.spearmanr(x, y, nan_policy='omit')
    ax = plt.gca()
    if p_rho < 0.0167:
        weight = 'bold'
    else:
        weight = 'normal'

    try:
        ax.annotate("rho = {:.2f}, p = {:.2e}".format(rho, p_rho),
                    xy=(.1, 1),
                    xycoords=ax.transAxes,
                    fontsize=8,
                    fontweight=weight)
        plt.axvline(x=0,
                    ymin=0,
                    ymax=5,
                    color='k',
                    ls='--',
                    lw=0.5)

    except ValueError:
        p_rho = float(ma.getdata(p_rho))
        ax.annotate("rho = {:.2f}, p = {:.2e}".format(rho, p_rho),
                    xy=(.1, 1),
                    xycoords=ax.transAxes,
                    fontsize=8,
                    fontweight=weight)

# plotting correlations
pdf = PdfPages("C:/Users/abiga\Box "
               "Sync\Abigail_Nicole\ChippiesTimeOfDay"
               "\TODCorrelations"
               "/SongVar_corr_wReg_JFORevisions.pdf")
sns.set(style='white',
        rc={"font.style": "normal",
            'lines.markersize': 1,
            'axes.labelsize': 5,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8})
sns.set_style('ticks')

g = sns.pairplot(data=combined_df,
                 x_vars=combined_df.columns[-1],
                 y_vars=combined_df.columns[1:5],
                 kind='reg')


g.map(corrfunc)
pdf.savefig(transparent=True)
pdf.close()
plt.show()