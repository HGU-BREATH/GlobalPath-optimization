import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import argparse

def main(inputfile, compare):
    data = pd.read_csv(inputfile)
    x = data['x_m'].values
    y = data['y_m'].values

    # data sampling with certain interval; you should find proper interval number.
    x_half = x[::10]
    y_half = y[::10]

    t_half = np.arange(len(x_half))

    dx = np.gradient(x_half, t_half)
    dy = np.gradient(y_half, t_half)

    ddx = np.gradient(dx, t_half)
    ddy = np.gradient(dy, t_half)

    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

    curvature[np.isclose(curvature, 0)] = 0

    print("Curvature:")
    for i in range(len(curvature)):
        print(f"{i} | x: {x_half[i]}, y: {y_half[i]}, curvature: {curvature[i]}")

    inflection_points = np.where(np.diff(np.sign(curvature)) != 0)[0]

    inflection_x = x_half[inflection_points]
    inflection_y = y_half[inflection_points]

    print("Inflection Points of Middle line:")
    for i in range(len(inflection_points)):
        print(f"x: {inflection_x[i]}, y: {inflection_y[i]}")

    data1 = pd.read_csv(compare)
    xc = data1['x'].values
    yc = data1['y'].values

    tree = KDTree(np.c_[xc, yc])
    distances, indices = tree.query(np.c_[inflection_x, inflection_y])

    inflection_xc = xc[indices]
    inflection_yc = yc[indices]

    print("Inflection Points of compare file (trajectory, wayoints, etc):")
    for i in range(len(indices)):
        print(f"x: {inflection_xc[i]}, y: {inflection_yc[i]}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', label='Middle Line', markersize=3)
    plt.plot(xc, yc, 'o-', label='Waypoint', markersize=3)
    plt.plot(xc, yc, 'o-', label='Trajectory', markersize=3)
    plt.plot(x_half, y_half, 'o-', label='Selected Data', markersize=3)
    plt.plot(inflection_x, inflection_y, 'rx', label='Inflection Points (Middle Line)', markersize=8)
    plt.plot(inflection_xc, inflection_yc, 'bx', label='Inflection Points (For compare)', markersize=8)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory with Inflection Points')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and save inflection points from trajectory data.')
    parser.add_argument('inputfile', type=str, help='Path to the input CSV file containing middle line data.')
    parser.add_argument('compare', type=str, help='Path to the input CSV file containing anoter data.')
    args = parser.parse_args()
    
    main(args.inputfile)