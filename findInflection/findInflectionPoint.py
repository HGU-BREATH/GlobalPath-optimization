import pandas as pd
import numpy as np
import scipy.signal as signal
import argparse
import os

def main(input_file):

    data = pd.read_csv(input_file)
    x = data['x_m'].values
    y = data['y_m'].values

    # data sampling with certain interval; you should find proper interval number.
    x_sample = x[::10]
    y_sample = y[::10]

    t_sample = np.arange(len(x_sample))

    dx = np.gradient(x_sample, t_sample)
    dy = np.gradient(y_sample, t_sample)

    ddx = np.gradient(dx, t_sample)
    ddy = np.gradient(dy, t_sample)

    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    for i in range(len(curvature)):
        if abs(curvature[i]) == 0:
            curvature[i] = 0

    print("Curvature:")
    for i in range(len(curvature)):
      print(f"{i} | x: {x_sample[i]}, y: {y_sample[i]}, culvature: {curvature[i]}")

    inflection_points = np.where(np.diff(np.sign(curvature)) != 0)[0]

    inflection_x = x_sample[inflection_points]
    inflection_y = y_sample[inflection_points]

    print("Inflection Points:")
    for i in range(len(inflection_points)):
        print(f"x: {inflection_x[i]}, y: {inflection_y[i]}")

    # '/home/{your directory}/f1tenth_Global_path_planning/src/inflectionPoints' 
    output_dir = '/home/onebean/f1tenth_Global_path_planning/src/inflectionPoints'  
    input_basename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, f"inflectionpoints_{input_basename}")

    os.makedirs(output_dir, exist_ok=True)

    inflection_points_df = pd.DataFrame({'x': inflection_x, 'y': inflection_y})
    inflection_points_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and save inflection points from trajectory data.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing middle line data.')
    args = parser.parse_args()
    
    main(args.input_file)
