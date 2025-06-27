# requirements & imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_senor_file
from data_proc_adarp import align_sensor_data, get_eda_data_around_tags, get_tag_timestamps, get_hr_data_around_tags, not_stressed_data_from_all_files
from filters import smooth_signal
from preprocessing import extract_eda_features, extract_hrv_features
from stress_models import train_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sensor_data1 = "/Users/austudent/Desktop/adarp_project/Part 101C/Sensor Data Day 1 csv files"

# starting small with eda and hr first so i don't get confused.
# will obvi add temp, acc, etc. once eda and hr work
 
eda_data1 = os.path.join(sensor_data1, "EDA.csv")
hr_data1 = os.path.join(sensor_data1, "HR.csv")
tag_data1 = os.path.join(sensor_data1, "tags.csv")
eda_df = load_senor_file(eda_data1)
hr_df = load_senor_file(hr_data1)
eda_df = align_sensor_data(eda_df, sampling_rate=4)
hr_df = align_sensor_data(hr_df, sampling_rate=1)
eda_df["EDA_Smooth"] = smooth_signal(eda_df['EDA'])

# tag times / time stamps

tag_times1 = get_tag_timestamps(tag_data1)
print(f"Number of stress/craving related event timestamps: {len(tag_times1)}")

eda_stress1 = get_eda_data_around_tags(sensor_data1, tag_times1, segment_size=60)
hr_stress1 = get_hr_data_around_tags(sensor_data1, tag_times1, segment_size=60)
eda_no_stress, hr_no_stress = not_stressed_data_from_all_files(data_folder=sensor_data1,
                                                               segment_length_to_skip=60*60,
                                                               include_eda=True,
                                                               include_temp=False,
                                                               include_acc=False,
                                                               include_bvp=False,
                                                               include_hr=True
                                                               
                                                               )


print(f'Stress segments: {len(eda_stress1)} and Non-stressed segments: {len(eda_no_stress)}')


def extract_features(eda_segments, hr_segments):
    eda_features = [ extract_eda_features(pd.Series(e), window_size=len(e))
                    for e in eda_segments]
    hrv_features = [ extract_hrv_features(pd.Series(h), window_size=len(h))
                    for h in hr_segments]
    return [pd.concat([e,h], axis=1) for e, h in zip(eda_features, hrv_features )]

stress_features = extract_features(eda_stress1, hr_stress1)
not_stressed_features = extract_features(eda_no_stress, hr_no_stress)
stress_df = pf.concat(stress_features).rest_index(drop=True)
not_stressed_df = extract_features(eda_no_stress, hr_no_stress)

stress_df['label'] = 1
not_stressed_df['label'] = 0
features_df = pd.concat([stress_df, not_stressed_df]).reset_index(drop=True)
print(f'Total feature samples:{features_df.shape[0]}')


x = features_df.drop("label", axis=1)
y = features_df["label"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2, random_state=22)
classifcation = RandomForestClassifier(random_state=22)
classifcation.fit(x_train, y_train)
y_prediction = classifcation.predict(x_test)

print(classification_report(y_test, y_prediction))


# GO BACK AND ADD EVERYTHING ELSE