import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns; sns.set()
from scipy.stats import ranksums, sem, levene
import csv
from matplotlib.ticker import FuncFormatter


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
            "_forTOD_SunriseTwilightNoon.csv"   
time_data = pd.DataFrame.from_csv(data_path, header=0, index_col=None)
# must remove duplicates -- have more than one bird from same recording -- duplicate catalog number and time data
time_data = time_data.drop_duplicates()

# combine tables using catalog no
combined_df = pd.merge(data_subset, time_data, on='CatalogNo')

# only keep ones with time data
combined_df = combined_df.drop(combined_df[combined_df.Sunrise == 
                                           '--'].index).copy().reset_index(
    drop=True)

print(combined_df.columns)
print(combined_df.shape)
print(combined_df.groupby('Sunrise').count()['CatalogNo'])
song_variables = combined_df.columns[1:5]

"""
Brown-Forsythe test (Levene's with median)
"""

with open("C:/Users/abiga/Box "
          "Sync/Abigail_Nicole/ChippiesTimeOfDay/TODDiscrete"
          "/SunriseCivilTwilightNoon_BrownForsythe_variances.csv",
          'wb') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerow(['Song Variable',
                         'Before After Sunrise p-value',
                         'Before Sunrise and After Noon p-value',
                         'After Sunrise and After Noon p-value',
                         'Before Sunrise Variance',
                         'Morning Variance',
                         'Afternoon Variance'
                         ])

    for sv in song_variables:
        before_sunrise = combined_df.loc[combined_df['Sunrise'] ==
                                         'before sunrise', sv]
        after_sunrise = combined_df.loc[combined_df['Sunrise'] ==
                                        'after sunrise', sv]
        sunrise_after_noon = combined_df.loc[combined_df['Sunrise'] ==
                                             'after noon', sv]

        filewriter.writerow([sv,
                             levene(before_sunrise, after_sunrise,
                                    center='median')[1],
                             levene(before_sunrise, sunrise_after_noon,
                                    center='median')[1],
                             levene(after_sunrise, sunrise_after_noon,
                                    center='median')[1],
                             np.var(before_sunrise),
                             np.var(after_sunrise),
                             np.var(sunrise_after_noon)
                             ])

"""
Wilcoxon Ranksums
"""

with open("C:/Users/abiga/Box "
          "Sync/Abigail_Nicole/ChippiesTimeOfDay/TODDiscrete"
          "/SunriseCivilTwilightNoon_WilcoxonRanksums_median.csv",
          'wb') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerow(['Song Variable',
                         'Before After Sunrise p-value',
                         'Before Sunrise and After Noon p-value',
                         'After Sunrise and After Noon p-value',
                         'Before Sunrise Median',
                         'Morning Median',
                         'Afternoon Median',
                         ])

    for sv in song_variables:
        before_sunrise = combined_df.loc[combined_df['Sunrise'] ==
                                         'before sunrise', sv]
        after_sunrise = combined_df.loc[combined_df['Sunrise'] ==
                                        'after sunrise', sv]
        sunrise_after_noon = combined_df.loc[combined_df['Sunrise'] ==
                                             'after noon', sv]

        filewriter.writerow([sv,
                             ranksums(before_sunrise, after_sunrise)[1],
                             ranksums(before_sunrise, sunrise_after_noon)[1],
                             ranksums(after_sunrise, sunrise_after_noon)[1],
                             np.exp(before_sunrise.median()),
                             np.exp(after_sunrise.median()),
                             np.exp(sunrise_after_noon.median()),
                             ])

""""
Box plots
"""

sv_titles = ['Duration of Song Bout (s)',
             'Mean Syllable Duration (ms)',
             'Mean Inter-Syllable Silence Duration (ms)',
             'Total Number of Syllables']

i = 0
for sv in song_variables:
    fig = plt.figure(figsize=(7, 11))
    sns.set(style='white')
    ax = sns.boxplot(x='Sunrise', y=sv, data=combined_df[['Sunrise', sv]], color='None',
                     fliersize=0, width=0.5, linewidth=2, order=['before sunrise', 'after sunrise', 'after noon'])
    ax = sns.stripplot(x='Sunrise', y=sv, data=combined_df[['Sunrise', sv]],
                       order=['before sunrise', 'after sunrise', 'after noon'],
                       palette=['black', '#95B2B8', '#F1D302'], size=7, jitter=True, lw=1, alpha=0.6)

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(sv_titles[i], fontsize=30)
    ax.set_xlabel('')
    ax.tick_params(labelsize=15, direction='out')
    plt.setp(ax.spines.values(), linewidth=2)
    if sv == 'Total Number of Syllables (log(number))':
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))
    elif sv == 'Duration of Song Bout (log(ms))':
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x)/1000)))
    else:
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))

    plt.savefig("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesTimeOfDay"
                "/TODDiscrete/" + sv + '_Sunrise' + '.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)
    i += 1


# box plot of entire population, not broken into time of day categories (for supplement)
i = 0
for sv in song_variables:
    fig = plt.figure(figsize=(3.5, 11))
    sns.set(style='white')
    sns.set_style("ticks")

    ax = sns.boxplot(y=sv, data=combined_df[[sv]], color='None',
                     fliersize=0, width=0.5, linewidth=2)
    ax = sns.stripplot(y=sv, data=combined_df[[sv]], color='grey', size=7, jitter=True, lw=1, alpha=0.6)

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(sv_titles[i], fontsize=30)
    ax.set_xlabel('')
    ax.tick_params(labelsize=15, direction='out')
    plt.setp(ax.spines.values(), linewidth=2)
    if sv == 'Total Number of Syllables (log(number))':
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))
    elif sv == 'Duration of Song Bout (log(ms))':
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x)/1000)))
    else:
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))

    plt.savefig("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesTimeOfDay"
                "/TODDiscrete/AllSongs_" + sv + '_Sunrise' + '.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)
    i += 1