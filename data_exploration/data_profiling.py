import pandas as pd
from ydata_profiling import ProfileReport
import os

folder_path = '../data/data_exploration'
df = pd.read_csv(os.path.join(folder_path, 'cars_train_preprocessed.csv'), index_col=0)

profile = ProfileReport(df, title="Cars Profiling Report")
profile.to_file(os.path.join(folder_path, "cars_train_preprocessed_report.html"))
