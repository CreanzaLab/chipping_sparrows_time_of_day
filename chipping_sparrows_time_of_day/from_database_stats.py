import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import seaborn as sns; sns.set()
import csv
from scipy.stats import ranksums

"""
Load data song data
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
combined_df['FromDatabase'] = combined_df['CatalogNo'].astype(str).str[0:2]

#divide up by database
fromXC = combined_df[combined_df.FromDatabase == 'XC']
fromML = combined_df[combined_df.FromDatabase != 'XC']

print(combined_df.shape)
print(fromXC.shape)
print(fromML.shape)
print(fromXC.Sunrise.unique())

""""
Wilcoxon Ranksums for 16 song variables ONLY FOR EAST SONGS
"""

fromXC_before = fromXC[fromXC.Sunrise == 'before sunrise']
fromML_before = fromML[fromML.Sunrise == 'before sunrise']

fromXC_morning = fromXC[fromXC.Sunrise == 'after sunrise']
fromML_morning = fromML[fromML.Sunrise == 'after sunrise']

fromXC_afternoon = fromXC[fromXC.Sunrise == 'after noon']
fromML_afternoon = fromML[fromML.Sunrise == 'after noon']

print(fromXC_before.shape)
print(fromML_before.shape)

print(fromXC_morning.shape)
print(fromML_morning.shape)

print(fromXC_afternoon.shape)
print(fromML_afternoon.shape)

with open("C:/Users/abiga/Box "
          "Sync/Abigail_Nicole/ChippiesTimeOfDay/TODXenoCantoQuality" 
          "/databaseBeforeOrMorningOnly_songProp_WilcoxonRanksums_XC_JFORevisions"
          ".csv",
          'wb') as \
        file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerow(['Song Variable',
                         'XC vs ML before statistic',
                         'XC vs ML morning statistic',
                         'XC vs ML afternoon statistic',
                         'XC vs ML before p-value',
                         'XC vs ML morning p-value',
                         'XC vs ML afternoon p-value'])

    for sv in combined_df.columns[1:5]:
        xc_b = np.asarray(fromXC_before[sv])
        ml_b = np.asarray(fromML_before[sv])
        xc_m = np.asarray(fromXC_morning[sv])
        ml_m = np.asarray(fromML_morning[sv])
        xc_a = np.asarray(fromXC_afternoon[sv])
        ml_a = np.asarray(fromML_afternoon[sv])
        filewriter.writerow([sv,
                             ranksums(xc_b, ml_b)[0],
                             ranksums(xc_m, ml_m)[0],
                             ranksums(xc_a, ml_a)[0],
                             ranksums(xc_b, ml_b)[1],
                             ranksums(xc_m, ml_m)[1],
                             ranksums(xc_a, ml_a)[1]
                             ])

