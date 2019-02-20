import pandas as pd

"""
Load data and organize/subset for correlations between 16 song variables and 
latitude, longitude, and year 
"""
data_path = 'C:/Users/abiga\Box ' \
            'Sync\Abigail_Nicole\ChippiesProject\FinalDataCompilation' \
            '/FinalDataframe_CombinedTables_withReChipper_' \
            'thenWithReExportedAs44100Hz_LogTransformed.csv'
log_song_data = pd.DataFrame.from_csv(data_path, header=0, index_col=None)

log_song_data_unique = log_song_data.loc[log_song_data[
    'ComparedStatus'].isin(['unique', 'use'])].copy().reset_index(drop=True)
print(log_song_data_unique.shape)

log_song_data_XCML = log_song_data_unique
log_song_data_XCML = log_song_data_unique[log_song_data_unique.FromDatabase
                                          != 'old']
print(log_song_data_XCML.shape)
print(log_song_data_XCML['FromDatabase'].unique())

data_for_TOD = log_song_data_XCML[['CatalogNo',
                                   'RecordingDay',
                                   'RecordingMonth',
                                   'RecordingTime',
                                   'BoutDuration_ms',
                                   'AvgSyllableDuration_ms',
                                   'AvgSilenceDuration_ms',
                                   'NumSyllables']]

data_for_TOD['RecordingTime'] = pd.to_datetime(data_for_TOD['RecordingTime'])
data_for_TOD['RecordingTime'] = [t.hour * 3600 + t.minute * 60 + t.second
                                 for t in data_for_TOD['RecordingTime']]

column_names = ['CatalogNo',
                'RecordingDay',
                'RecordingMonth',
                'RecordingTime',
                'Duration of Song Bout (log(ms))',
                'Mean Syllable Duration (log(ms))',
                'Mean Inter-Syllable Silence Duration (log(ms))',
                'Total Number of Syllables (log(number))']
data_for_TOD.columns = column_names

data_for_TOD.to_csv("C:/Users/abiga/Box "
                    "Sync/Abigail_Nicole/ChippiesTimeOfDay"
                    "/FinalChippiesDataReExportedAs44100Hz_LogTransformed"
                    "_forTOD.csv")
