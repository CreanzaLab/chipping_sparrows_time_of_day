from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

col_to_skip = ['Longitude',
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

combined_df.CatalogNo.to_csv("C:/Users/abiga\Box "
                   "Sync\Abigail_Nicole\ChippiesTimeOfDay\RecordingInfo"
                   "/TimeOfDayPaper_RecordingsUsed.csv", index=False)

combined_df_all_months = combined_df.copy()

# remove month 8 since it only has one recording
combined_df = combined_df[combined_df['RecordingMonth'] != 8.0]

song_variables = ['Duration of Song Bout (log(ms))', 'Total Number of '
                                                     'Syllables (log(number))']

lower = combined_df[combined_df.Latitude <= 35]
middle = combined_df[(combined_df.Latitude > 35) & (combined_df.Latitude < 45)]
upper = combined_df[combined_df.Latitude >= 45]


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

# lower latitudes
rs = np.zeros((5, 5))

for sv in song_variables:
    for i in range(len(months)-1):
        for j in range(i+1, len(months)):
            m1 = lower.loc[lower['RecordingMonth'] == months[i], sv]
            m2 = lower.loc[lower['RecordingMonth'] == months[j], sv]
            rs[j, i] = ranksums(m1, m2)[1]
    results = pd.DataFrame(data=rs, index=months)
    results.to_csv("C:/Users/abiga\Box "
                   "Sync\Abigail_Nicole\ChippiesTimeOfDay\MonthDiscrete"
                   "/Wilcoxon_" + sv + "_lower.csv", header=months)

# middle latitudes
rs = np.zeros((5, 5))

for sv in song_variables:
    for i in range(len(months)-1):
        for j in range(i+1, len(months)):
            m1 = middle.loc[middle['RecordingMonth'] == months[i], sv]
            m2 = middle.loc[middle['RecordingMonth'] == months[j], sv]
            rs[j, i] = ranksums(m1, m2)[1]
    results = pd.DataFrame(data=rs, index=months)
    results.to_csv("C:/Users/abiga\Box "
                   "Sync\Abigail_Nicole\ChippiesTimeOfDay\MonthDiscrete"
                   "/Wilcoxon_" + sv + "_middle.csv", header=months)

# upper latitudes
rs = np.zeros((5, 5))

for sv in song_variables:
    for i in range(len(months)-1):
        for j in range(i+1, len(months)):
            m1 = upper.loc[upper['RecordingMonth'] == months[i], sv]
            m2 = upper.loc[upper['RecordingMonth'] == months[j], sv]
            rs[j, i] = ranksums(m1, m2)[1]
    results = pd.DataFrame(data=rs, index=months)
    results.to_csv("C:/Users/abiga\Box "
                   "Sync\Abigail_Nicole\ChippiesTimeOfDay\MonthDiscrete"
                   "/Wilcoxon_" + sv + "_upper.csv", header=months)


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

# lower latitudes
i = 0
for sv in song_variables:
    fig = plt.figure(figsize=(7, 11))
    sns.set(style='white')

    ax = sns.boxplot(x='RecordingMonth', y=sv, data=lower[[
        'RecordingMonth', sv]], color='None', fliersize=0, width=0.5,
                     linewidth=2)
    ax = sns.stripplot(x='RecordingMonth', y=sv, data=lower[[
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
        ax.set(ylim=(np.log(4), np.log(165)))
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))
    elif sv == 'Duration of Song Bout (log(ms))':
        ax.set(ylim=(np.log(0.3 * 1000), np.log(8.1 * 1000)))
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x)/1000)))
    else:
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))

    plt.savefig("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesTimeOfDay"
                "/MonthDiscrete/" + "MonthBoxplot_" + sv + '_lower.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)

    i += 1

# middle latitudes
i = 0
for sv in song_variables:
    fig = plt.figure(figsize=(7, 11))
    sns.set(style='white')

    ax = sns.boxplot(x='RecordingMonth', y=sv, data=middle[[
        'RecordingMonth', sv]], color='None', fliersize=0, width=0.5,
                     linewidth=2)
    ax = sns.stripplot(x='RecordingMonth', y=sv, data=middle[[
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
        ax.set(ylim=(np.log(4), np.log(165)))
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))
    elif sv == 'Duration of Song Bout (log(ms))':
        ax.set(ylim=(np.log(0.3 * 1000), np.log(8.1 * 1000)))
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x)/1000)))
    else:
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))

    plt.savefig("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesTimeOfDay"
                "/MonthDiscrete/" + "MonthBoxplot_" + sv + '_middle.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)

    i += 1

# upper latitudes
i = 0
for sv in song_variables:
    fig = plt.figure(figsize=(7, 11))
    sns.set(style='white')

    ax = sns.boxplot(x='RecordingMonth', y=sv, data=upper[[
        'RecordingMonth', sv]], color='None', fliersize=0, width=0.5,
                     linewidth=2)
    ax = sns.stripplot(x='RecordingMonth', y=sv, data=upper[[
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
        ax.set(ylim=(np.log(4), np.log(165)))
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))
    elif sv == 'Duration of Song Bout (log(ms))':
        ax.set(ylim=(np.log(0.3 * 1000), np.log(8.1 * 1000)))
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x)/1000)))
    else:
        ax.get_yaxis().set_major_formatter(FuncFormatter(
            lambda x, p: "%.1f" % (np.exp(x))))

    plt.savefig("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesTimeOfDay"
                "/MonthDiscrete/" + "MonthBoxplot_" + sv + '_upper.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)

    i += 1

"""
Scatter Plot song duration vs date (color in latitude)
"""

combined_df_all_months = combined_df_all_months.dropna(axis=0, subset=['RecordingDay'])

# use month and day columns to turn into a datetime to be able to plot with date as x-axis
combined_df_all_months['Datetime'] = pd.to_datetime(combined_df_all_months[['RecordingDay', 'RecordingMonth']].astype(int)
                                         .astype(str).apply(' '.join, 1), format='%d %m')
pydatetime = combined_df_all_months['Datetime'].dt.to_pydatetime()
combined_df_all_months['DateAsNum'] = mdates.date2num(pydatetime)


fig = plt.figure(figsize=(11.69, 8.27))
my_dpi = 96
sns.set(style='white')
sns.set_style("ticks")

ax = sns.scatterplot(x=combined_df_all_months['DateAsNum'], y=combined_df_all_months[song_variables[0]],
                     hue=combined_df_all_months['Latitude'], linewidth=0)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())
monthFmt = mdates.DateFormatter("%b")
ax.xaxis.set_major_formatter(monthFmt)
ax.set(ylim=(np.log(0.3 * 1000), np.log(8.1 * 1000)))
ax.get_yaxis().set_major_formatter(FuncFormatter(
    lambda x, p: "%.1f" % (np.exp(x) / 1000)))

plt.savefig("C:/Users/abiga\Box Sync\Abigail_Nicole\ChippiesTimeOfDay"
            "/MonthDiscrete/" + "MonthBoxplot_" + 'SongDuration' + '_latitude2.pdf', type='pdf',
            dpi=fig.dpi, bbox_inches='tight',
            transparent=True)
plt.show()
plt.close()
