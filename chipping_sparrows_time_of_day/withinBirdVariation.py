from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns; sns.set()
from scipy.stats import ranksums, levene
import csv
import numpy as np
from matplotlib.ticker import FuncFormatter


"""
Load both song and sunrise data and organize/subset by before sunrise and month
"""
# load in song data
data_path = "C:/Users/abiga/Box " \
            "Sync/Abigail_Nicole/ChippiesTimeOfDay" \
            "/WithinBirdVariation\WithinBirdVariationGzips\AnalysisOutput_forAllWithinVariationGzips.csv"
log_song_data = pd.DataFrame.from_csv(data_path, header=0, index_col=None)
log_song_data['CatalogNo'] = log_song_data['FileName'].str.split('_', expand=True)[1]

# load in time data --> before or after sunrise, twilights, and noon (only going to use sunrise and noon)
data_path = "C:/Users/abiga/Box " \
            "Sync/Abigail_Nicole/ChippiesTimeOfDay" \
            "/FinalChippiesDataReExportedAs44100Hz_LogTransformed" \
            "_forTOD_SunriseTwilightNoon.csv"
time_data = pd.DataFrame.from_csv(data_path, header=0, index_col=None)
# must remove duplicates -- have more than one bird from same recording -- duplicate catalog number and time data
time_data = time_data.drop_duplicates()

# combine tables using catalog no
combined_df = pd.merge(log_song_data, time_data, on='CatalogNo')
print(combined_df.groupby('CatalogNo').count())


"""
Box Plots
"""

order = ['XC137480',
         'XC125194',
         '61915841',
         '62503991',
         'XC131641',
         '132218',
         '48237091',
         '47979571',
         'XC131639',
         'XC253974',
         '73955',
         '76777',
         'XC77992',
         'XC269237',
         '29810211']

song_variables = ['bout_duration(ms)', 'num_syllables']
combined_df[song_variables[0]] = combined_df[song_variables[0]].apply(np.log)
combined_df[song_variables[1]] = combined_df[song_variables[1]].apply(np.log)

sv_titles = ['Duration of Song Bout (s)',
             'Total Number of Syllables']

i = 0
for sv in song_variables:
    fig = plt.figure(figsize=(20, 10))
    sns.set(style='white')
    sns.set_style("ticks")

    ax = sns.boxplot(x='CatalogNo', y=sv, data=combined_df[[
        'CatalogNo', sv]], color='None', fliersize=0, width=0.5,
                     linewidth=2, order=order)
    ax = sns.stripplot(x='CatalogNo', y=sv, data=combined_df[[
        'CatalogNo', sv]], size=7, jitter=True, lw=1, alpha=0.6,
                       color='grey', order=order)

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(sv_titles[i], fontsize=30)
    ax.tick_params(labelsize=30, direction='out')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.setp(ax.spines.values(), linewidth=2)

    if sv == 'num_syllables':
        ax.set(ylim=(np.log(4), np.log(165)))
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))
    elif sv == 'bout_duration(ms)':
        ax.set(ylim=(np.log(0.3 * 1000), np.log(8.1 * 1000)))
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x)/1000)))
    else:
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))

    plt.savefig("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesTimeOfDay"
                "/WithinBirdVariation/WithinBirdVariation_" + sv + '.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)

    plt.show()

    i += 1
