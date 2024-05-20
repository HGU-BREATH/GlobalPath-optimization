import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# 데이터 로드
data = pd.read_csv('./src/midLine/setBreath.csv')
x = data['x_m'].values
y = data['y_m'].values

# 데이터 샘플링 (간격 10으로 선택)
x_half = x[::10]
y_half = y[::10]

# t 값 생성: 유클리드 거리 기반으로 생성
t_half = np.arange(len(x_half))

# 1차 도함수 계산
dx = np.gradient(x_half, t_half)
dy = np.gradient(y_half, t_half)

# 2차 도함수 계산
ddx = np.gradient(dx, t_half)
ddy = np.gradient(dy, t_half)

# 곡률 계산
curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

# 곡률이 0이 되는 문제 해결
curvature[np.isclose(curvature, 0)] = 0

print("Curvature:")
for i in range(len(curvature)):
    print(f"{i} | x: {x_half[i]}, y: {y_half[i]}, curvature: {curvature[i]}")

# 변곡점 찾기: 곡률의 부호 변화
inflection_points = np.where(np.diff(np.sign(curvature)) != 0)[0]

# 변곡점 위치 확인
inflection_x = x_half[inflection_points]
inflection_y = y_half[inflection_points]

# 결과 출력
print("Inflection Points:")
for i in range(len(inflection_points)):
    print(f"x: {inflection_x[i]}, y: {inflection_y[i]}")

# 비교를 위한 다른 trajectory 데이터 로드
data1 = pd.read_csv('/home/onebean/f1tenth_Global_path_planning/src/trajectory/trajectory1.csv')
xc = data1['x'].values
yc = data1['y'].values

# KDTree를 사용하여 가장 가까운 포인트 찾기
tree = KDTree(np.c_[xc, yc])
distances, indices = tree.query(np.c_[inflection_x, inflection_y])

# another trajectory에서의 변곡점 위치 확인
inflection_xc = xc[indices]
inflection_yc = yc[indices]

# 비교를 위한 다른 trajectory 데이터 로드
data2 = pd.read_csv('/home/onebean/f1tenth_Global_path_planning/src/waypoints/waypoint1.csv')
xw = data2['x'].values
yw = data2['y'].values

# KDTree를 사용하여 가장 가까운 포인트 찾기
tree = KDTree(np.c_[xw, yw])
distances, indices = tree.query(np.c_[inflection_x, inflection_y])

# another trajectory에서의 변곡점 위치 확인
inflection_xw = xw[indices]
inflection_yw = yw[indices]

# 결과 출력
print("Another Trajectory Inflection Points:")
for i in range(len(indices)):
    print(f"x: {inflection_xw[i]}, y: {inflection_yw[i]}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', label='Middle Line', markersize=3)
plt.plot(xw, yw, 'o-', label='Waypoint', markersize=3)
plt.plot(xc, yc, 'o-', label='Trajectory', markersize=3)
plt.plot(x_half, y_half, 'o-', label='Selected Data', markersize=3)
plt.plot(inflection_x, inflection_y, 'rx', label='Inflection Points (Middle Line)', markersize=8)
plt.plot(inflection_xw, inflection_yw, 'bx', label='Inflection Points (Waypoint)', markersize=8)
plt.plot(inflection_xc, inflection_yc, 'bx', label='Inflection Points (Trajectory)', markersize=8)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectory with Inflection Points')
plt.legend()
plt.grid(True)
plt.show()
