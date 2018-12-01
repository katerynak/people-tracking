#!/usr/bin/python3

import pandas as pd

"""
count a distance only of correspondent ids present in both ground truth and estimation
"""

selected_ids = [10, 36, 42]

for id in selected_ids:
    avg_dist = 0
    i = 0

    truth_data = pd.read_csv("../selected_pedestrians_data/groundtruth_{}.csv".format(id), header=None)[[0, 2, 3]]
    truth_data.columns = ['frame', 'tx', 'ty']
    estimated_data = pd.read_csv("../selected_pedestrians_data/pedestrian{}.csv".format(id), header=None)
    estimated_data.columns = ['frame', 'ex', 'ey']

    combined_data = pd.merge(truth_data, estimated_data, how='inner', on=['frame'])

    combined_data['distances'] = ((combined_data['ex'] - combined_data['tx'])**2 + (combined_data['ey']
                                                                                    - combined_data['ty'])**2)**(0.5)

    dist_avg = combined_data['distances'].mean()

    frames =len(combined_data['frame'])
    total_frames  = len(truth_data['frame'])
    estimated_frames = len(estimated_data['frame'])
    print("ID: {} average_distance: {}, total frames: {}"
          ", missed frames: {}".format(id, dist_avg, total_frames, total_frames - frames))

    print(set(truth_data['frame']) - set(combined_data['frame']))
