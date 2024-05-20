import pandas as pd
import numpy as np
import scipy.signal as signal
import argparse
import os

def main(input_file):

    # 입력 파일 로드
    data = pd.read_csv(input_file)
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

    # 출력 디렉토리 및 파일 경로 설정
    output_dir = '/home/onebean/f1tenth_Global_path_planning/src/inflectionPoints/'  # 여기에 출력 디렉토리를 설정
    input_basename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, f"inflectionpoints_{input_basename}")

    # 출력 디렉토리가 없는 경우 생성
    os.makedirs(output_dir, exist_ok=True)

    # 변곡점 데이터를 CSV 파일로 저장
    inflection_points_df = pd.DataFrame({'x': inflection_x, 'y': inflection_y})
    inflection_points_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and save inflection points from trajectory data.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing trajectory data.')
    args = parser.parse_args()
    
    main(args.input_file)
