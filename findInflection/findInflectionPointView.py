import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import argparse

def main(inputfile):

    data = pd.read_csv(inputfile)
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

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', label='Original Data', markersize=3)
    plt.plot(x_sample, y_sample, 'o-', label='Selected Data', markersize=3)
    plt.plot(inflection_x, inflection_y, 'rx', label='Inflection Points', markersize=8)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory with Inflection Points')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and show inflection points from trajectory data.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing middle line data.')
    args = parser.parse_args()
    
    main(args.input_file)

