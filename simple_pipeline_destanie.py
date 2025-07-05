# requirements & imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_sensor_file
from data_proc_adarp import align_sensor_data, get_eda_data_around_tags, get_tag_timestamps, get_hr_data_around_tags, not_stressed_data_from_all_files, get_temp_data_around_tags, get_bvp_data_around_tags, get_acc_data_around_tags
from filters import smooth_signal
from preprocessing import extract_eda_features, extract_hrv_features, extract_temp_features, extract_acc_features, extract_bvp_features
from stress_models import train_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


sensor_data1 = "/Users/austudent/Desktop/adarp_project/Part 101C/Sensor Data Day 1 csv files"

# starting small with eda and hr first so i don't get confused.
# will obvi add temp, acc, etc. once eda and hr work
 
eda_data1 = os.path.join(sensor_data1, "EDA.csv")
hr_data1 = os.path.join(sensor_data1, "HR.csv")
tag_data1 = os.path.join(sensor_data1, "tags.csv")
bvp_data1 = os.path.join(sensor_data1, "BVP.csv")
temp_data1 = os.path.join(sensor_data1, "TEMP.csv")
acc_data1 = os.path.join(sensor_data1, "ACC.csv")
eda_df = load_sensor_file(eda_data1)
hr_df = load_sensor_file(hr_data1)
eda_df = align_sensor_data(eda_df, sampling_rate=4)
hr_df = align_sensor_data(hr_df, sampling_rate=1)
eda_df["EDA_Smooth"] = smooth_signal(eda_df['EDA'])

# tag times / time stamps

tag_times1 = get_tag_timestamps(tag_data1)
print(f"Number of stress/craving related event timestamps: {len(tag_times1)}")

eda_stress1 = get_eda_data_around_tags(sensor_data1, tag_times1, segment_size=60)
hr_stress1 = get_hr_data_around_tags(sensor_data1, tag_times1, segment_size=60)
temp_stress1 = get_temp_data_around_tags(temp_data1, tag_times1)
bvp_stress1 = get_bvp_data_around_tags(bvp_data1, tag_times1)
acc_stress1 = get_acc_data_around_tags(acc_data1, tag_times1)
temp_no_stress = get_temp_data_around_tags(temp_data1, tag_times1)
bvp_no_stress = get_bvp_data_around_tags(bvp_data1, tag_times1)
acc_no_stress = get_acc_data_around_tags(acc_data1, tag_times1)
eda_no_stress, hr_no_stress, temp_no_stress, acc_no_stress, bvp_no_stress = not_stressed_data_from_all_files(
    data_folder=sensor_data1,
    segment_length_to_skip=60*60,
    include_eda=True,
    include_hr=True,
    include_temp=True,
    include_acc=True,
    include_bvp=True
)
print(f'EDA segments around stress events: {len(eda_stress1)}')
print(f'HR segments around stress events: {len(hr_stress1)}')
print(f'TEMP segments around stress events: {len(temp_stress1)}')
print(f'BVP segments around stress events: {len(bvp_stress1)}')
print(f'ACC segments around stress events: {len(acc_stress1)}')   


print(f'Stress segments: {len(eda_stress1)} and Non-stressed segments: {len(eda_no_stress)}')


def extract_features(eda_segments, hr_segments, temp_segments, acc_segments, bvp_segments):
    all_features = []
    for eda, hr, temp, acc, bvp in zip(eda_segments, hr_segments, temp_segments, acc_segments, bvp_segments):
        eda_feat = extract_eda_features(pd.Series(eda))
        hrv_feat = extract_hrv_features(pd.Series(hr))
        temp_feat = extract_temp_features(pd.Series(temp))
        acc_feat = extract_acc_features(pd.DataFrame(acc))
        bvp_feat = extract_bvp_features(pd.Series(bvp))
        all_feats = pd.concat([eda_feat, hrv_feat, temp_feat, acc_feat, bvp_feat], axis=1)
        all_features.append(all_feats)
    return all_features


stress_features = extract_features(eda_stress1, hr_stress1, temp_stress1, acc_stress1, bvp_stress1)
not_stressed_features = extract_features(eda_no_stress, hr_no_stress, temp_no_stress, acc_no_stress, bvp_no_stress)
stress_df = pd.concat(stress_features).reset_index(drop=True)
not_stressed_df = pd.concat(not_stressed_features).reset_index(drop=True)

stress_df['label'] = 1
not_stressed_df['label'] = 0
features_df = pd.concat([stress_df, not_stressed_df]).reset_index(drop=True)
print(f'Total feature samples:{features_df.shape[0]}')


x = features_df.drop("label", axis=1)
y = features_df["label"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2, random_state=22)
classification = RandomForestClassifier(random_state=22)
classification.fit(x_train, y_train)
y_prediction = classification.predict(x_test)

print(classification_report(y_test, y_prediction))


# GO BACK AND ADD EVERYTHING ELSE

conf_matrix = confusion_matrix(y_test, y_prediction)
print(conf_matrix)