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
            "/FinalChippiesDataReExportedAs44100Hz_LogTransformed_forTOD.csv"
log_song_data = pd.DataFrame.from_csv(data_path, header=0, index_col=None)

col_to_skip = ['Latitude', 'Longitude', 'RecordingDay',
               'RecordingYear', 'RecordingTime',
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
print(combined_df.shape)  # 314
print(combined_df.RecordingMonth.unique())


# remove month 8 since it only has one recording
combined_df = combined_df[combined_df['RecordingMonth'] != 8.0]

song_variables = ['Duration of Song Bout (log(ms))', 'Total Number of '
                                                     'Syllables (log(number))']
print(combined_df.groupby('RecordingMonth').count())

combined_df.CatalogNo.to_csv("C:/Users/abiga\Box "
                   "Sync\Abigail_Nicole\ChippiesTimeOfDay\RecordingInfo"
                   "/TimeOfDayPaper_RecordingsUsed.csv", index=False)

"""
Variance in each month
"""

months = sorted(combined_df.RecordingMonth.unique())

with open("C:/Users/abiga\Box "
          "Sync\Abigail_Nicole\ChippiesTimeOfDay\MonthDiscrete"
          "/Variance_ofMonths.csv",
          'wb') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerow(['Song Variable',
                         'March',
                         'April',
                         'May',
                         'June',
                         'July'
                         ])

    for sv in song_variables:
        variance = []
        for m in months:
            vector = combined_df.loc[combined_df['RecordingMonth'] == m, sv]
            variance.append(np.var(vector))
        filewriter.writerow([sv] + variance)


"""
Brown-Forsythe test (Levene's with median)
"""
# between all months
values_per_group = [col for col_name, col in combined_df.groupby(
    'RecordingMonth')['Duration of Song Bout (log(ms))']]
print(levene(*values_per_group, center='median'))


months = sorted(combined_df.RecordingMonth.unique())

rs = np.zeros((5, 5))

for sv in song_variables:
    for i in range(len(months)-1):
        for j in range(i+1, len(months)):
            m1 = combined_df.loc[combined_df['RecordingMonth'] == months[i], sv]
            m2 = combined_df.loc[combined_df['RecordingMonth'] == months[j], sv]
            rs[j, i] = levene(m1, m2, center='median')[1]
    results = pd.DataFrame(data=rs, index=months)
    results.to_csv("C:/Users/abiga\Box "
                   "Sync\Abigail_Nicole\ChippiesTimeOfDay\MonthDiscrete"
                   "/BrownForsythe_" + sv + ".csv", header=months)


"""
Wilcoxon Ranksums
"""
months = sorted(combined_df.RecordingMonth.unique())

rs = np.zeros((5, 5))

for sv in song_variables:
    for i in range(len(months)-1):
        for j in range(i+1, len(months)):
            m1 = combined_df.loc[combined_df['RecordingMonth'] == months[i], sv]
            m2 = combined_df.loc[combined_df['RecordingMonth'] == months[j], sv]
            rs[j, i] = ranksums(m1, m2)[1]
    results = pd.DataFrame(data=rs, index=months)
    results.to_csv("C:/Users/abiga\Box "
                   "Sync\Abigail_Nicole\ChippiesTimeOfDay\MonthDiscrete"
                   "/Wilcoxon_" + sv + ".csv", header=months)

""""
Box plots
"""

sv_titles = ['Duration of Song Bout (s)',
             'Total Number of Syllables']

# box plot for duration of song bout, take exponential (and convert from ms to s for bout duration)
i = 0
for sv in song_variables:
    fig = plt.figure(figsize=(7, 11))
    sns.set(style='white')

    ax = sns.boxplot(x='RecordingMonth', y=sv, data=combined_df[[
        'RecordingMonth', sv]], color='None', fliersize=0, width=0.5,
                     linewidth=2)
    ax = sns.stripplot(x='RecordingMonth', y=sv, data=combined_df[[
        'RecordingMonth', sv]], size=7, jitter=True, lw=1, alpha=0.6,
                       color='grey')

    # Make the boxplot fully transparent
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0))

    ax.set_ylabel(sv_titles[i], fontsize=30)
    ax.tick_params(labelsize=30, direction='out')
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
                "/MonthDiscrete/" + "MonthBoxplot_" + sv + '.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)

    i += 1
