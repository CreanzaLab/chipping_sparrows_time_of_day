import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns; sns.set()
from scipy.stats import ranksums
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
print(combined_df.groupby('Sunrise').count()['CatalogNo'])

song_variables = combined_df.columns[1:5]

"""
Wilcoxon Ranksums
"""

with open("C:/Users/abiga/Box "
          "Sync/Abigail_Nicole/ChippiesTimeOfDay/TODDiscrete"
          "/SunriseCivilTwilightNoon_WilcoxonRanksums.csv",
          'wb') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerow(['Song Variable',
                         'Before After Sunrise p-value',
                         'Before Sunrise and After Noon p-value',
                         'After Sunrise and After Noon p-value'
                         ])

    for sv in song_variables:
        before_sunrise = combined_df.loc[combined_df['Sunrise'] ==
                                         'before sunrise', sv]
        after_sunrise = combined_df.loc[combined_df['Sunrise'] ==
                                        'after sunrise', sv]
        sunrise_after_noon = combined_df.loc[combined_df['Sunrise'] ==
                                             'after noon', sv]

        filewriter.writerow([sv, ranksums(before_sunrise, after_sunrise)[1],
                             ranksums(before_sunrise, sunrise_after_noon)[1],
                             ranksums(after_sunrise, sunrise_after_noon)[1]
                             ])

""""
Box plots (change out the sv, the index for title, and the formatting for the two different box plot sets - bout 
duration and number of syllables)
"""

sv_titles = ['Duration of Song Bout (s)',
             'Mean Syllable Duration (ms)',
             'Mean Inter-Syllable Silence Duration (ms)',
             'Total Number of Syllables']

# box plot for duration of song bout, take exponential (and convert from ms to s for bout duration)
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
    ax.tick_params(labelsize=30, direction='out')
    ax.set(xticklabels=[])
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


    # plt.tight_layout()
    # pdf = PdfPages("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesProject\StatsOfFinalData_withReChipperReExported/TimeAnalysis"
    #                "/PaperVersion/" + sv + '_Sunrise' + '.pdf')
    # pdf.savefig(orientation='landscape')
    # pdf.close()
    # plt.show()

    plt.savefig("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesTimeOfDay"
                "/TODDiscrete/" + sv + '_Sunrise' + '.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)
    i += 1

