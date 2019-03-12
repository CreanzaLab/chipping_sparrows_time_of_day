import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from ifdvsonogramonly import ifdvsonogramonly
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns; sns.set()
from matplotlib.ticker import FuncFormatter


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

song_variables = combined_df.columns[1:5]

"""
Finding an example of East, West, South songs (the ones closes to the average for specified song features of interest)
"""
var_of_interest = ['Duration of Song Bout (log(ms))', 'Total Number of '
                                                      'Syllables (log(number))']
var_diffs = ['DiffBoutDur', 'DiffSyllDur', 'DiffSilenceDur', 'DiffNumSylls']

example_files = {}
bout_dur = {}

for time in ['before sunrise', 'after sunrise', 'after noon']:
    mean_df = pd.DataFrame(columns=['CatalogNo', 'DiffBoutDur', 'DiffSyllDur', 'DiffSilenceDur', 'DiffNumSylls'])
    for i in range(0, 2):
        tod_data = combined_df.loc[combined_df['Sunrise'] == time]
        mean_df['CatalogNo'] = tod_data['CatalogNo']
        mean_df['Duration of Song Bout (log(ms))'] = tod_data['Duration of ' \
                                                              'Song Bout (' \
                                                              'log(ms))']
        mean_df[var_diffs[i]] = abs(tod_data[var_of_interest[i]] - tod_data[
            var_of_interest[i]].mean())
    mean_df['DiffSum'] = mean_df[var_diffs].sum(axis=1)
    example_files.update({time: mean_df.loc[mean_df['DiffSum'].idxmin()][
        'CatalogNo']})
    bout_dur.update({time: mean_df.loc[mean_df['DiffSum'].idxmin()][
        'Duration of Song Bout (log(ms))']})
    del mean_df

print(example_files)

"""
Load in example songs and make figures
"""
song_names = ['176261_44k_b5of11_beforesunriseExt.wav',
              'XC76506_b1of2_morningExt.wav',
              '76777_b4of17_afternoonExt.wav']
for name in song_names:
    song_file = "C:/Users/abiga\Box " \
                "Sync\Abigail_Nicole\ChippiesTimeOfDay" \
                "\TODExampleSongs_boutDurNumSylls/ExtendedTimeOfRecording/" +\
                name
    song, rate = sf.read(song_file)
    sonogram, timeAxis_conversion, freqAxis_conversion = ifdvsonogramonly(song,
                                                                          rate,
                                                                          1024,
                                                                          1010.0,
                                                                          2.0)
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(1, 1, 1)
    # sns.set(style='white')
    [rows, cols] = np.shape(sonogram)
    im = plt.imshow(np.log(sonogram+3),
                    cmap='gray_r',
                    extent=[0, cols, 0, rows],
                    aspect='auto')

    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(
            lambda x, p: "%.2f" % (x*timeAxis_conversion/1000)))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(
            lambda x, p: "%.0f" % (x*freqAxis_conversion/1000)))
    plt.tick_params(labelsize=14)
    plt.savefig("C:/Users/abiga\Box "
                "Sync/Abigail_Nicole/ChippiesTimeOfDay"
                "/TODExampleSongs_boutDurNumSylls/ExtendedTimeOfRecording/" +
                name + '_sonogram' + '.pdf', type='pdf',
                dpi=fig.dpi, bbox_inches='tight',
                transparent=True)
    # plt.show()
