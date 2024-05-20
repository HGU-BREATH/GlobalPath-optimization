import numpy as np
import pandas as pd
import scipy.interpolate as interp
import sys
import os

def find_new_start_point(points, start_point, threshold):
    distances = np.linalg.norm(points - start_point, axis=1)
    new_start_point_index = np.where(distances > threshold)[0][0]
    return new_start_point_index - 1

def find_closest_point(points, start_point_index, search_offset):
    start_point = points[start_point_index]
    distances = np.linalg.norm(points - start_point, axis=1)
    closest_point_index = np.argmin(distances[search_offset + start_point_index + 1:]) + (search_offset + start_point_index + 1)
    return closest_point_index

def main(input_file):
    trajectory = pd.read_csv(input_file)

    x = trajectory['x'].to_numpy()
    y = trajectory['y'].to_numpy()
    v = trajectory['v'].to_numpy()
    points = np.column_stack((x, y, v))

    initial_start_point = points[0]
    search_offset = int(len(points) / 10)

    threshold = 0.05
    new_start_point_index = find_new_start_point(points, initial_start_point, threshold)
    closest_point_index = find_closest_point(points, new_start_point_index, search_offset)

    trimmed_trajectory = trajectory.iloc[new_start_point_index:closest_point_index + 1]
    trimmed_trajectory.loc[trimmed_trajectory.index[-1], ['x', 'y', 'v']] = trimmed_trajectory.iloc[0][['x', 'y', 'v']]
    trimmed_trajectory.loc[trimmed_trajectory['v'] == 0, 'v'] = 1
    trimmed_trajectory['t'] = trimmed_trajectory['t'] - trimmed_trajectory['t'][0]
    trimmed_trajectory['t'] = trimmed_trajectory['t'].round(5)

    output_dir = '/home/onebean/GlobalPath-optimization/src/trajectory'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{os.path.basename(input_file)}")

    trimmed_trajectory.to_csv(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py input.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    main(input_file)

