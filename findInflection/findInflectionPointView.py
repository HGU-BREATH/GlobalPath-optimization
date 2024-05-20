import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('./src/midLine/mid_breath.csv')
x = data['x_m'].values
y = data['y_m'].values

# Choose data with interval 16
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
for i in range(len(curvature)):
    if abs(curvature[i]) == 0:
        curvature[i] = 0

print("Curvature:")
for i in range(len(curvature)):
    print(f"{i} | x: {x_half[i]}, y: {y_half[i]}, culvature: {curvature[i]}")


# 변곡점 찾기: 곡률의 부호 변화
inflection_points = np.where(np.diff(np.sign(curvature)) != 0)[0]

# 변곡점 위치 확인
inflection_x = x_half[inflection_points]
inflection_y = y_half[inflection_points]

# 결과 출력
print("Inflection Points:")
for i in range(len(inflection_points)):
    print(f"x: {inflection_x[i]}, y: {inflection_y[i]}")

# For comparison
# data1 = pd.read_csv('/home/onebean/f1tenth_Global_path_planning/src/optWaypoints/optWaypoint.csv')
# xc = data1['x'].values
# yc = data1['y'].values


# 시각화
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', label='Original Data', markersize=3)
# plt.plot(xc, yc, 'o-', label='another trajecoty', markersize=3)
plt.plot(x_half, y_half, 'o-', label='Selected Data', markersize=3)
plt.plot(inflection_x, inflection_y, 'rx', label='Inflection Points', markersize=8)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectory with Inflection Points')
plt.legend()
plt.grid(True)
plt.show()
