import matplotlib.pyplot as plt
import numpy as np
import sys

def main(file1, file2, file3, file4):
    
    data1 = np.genfromtxt(file1, delimiter=',', skip_header=1)  # 헤더가 있다면 skip_header=1을 사용
    x1 = data1[:, 1] 
    y1 = data1[:, 2] 

    data2 = np.genfromtxt(file2, delimiter=',', skip_header=1)
    x2 = data2[:, 1] 
    y2 = data2[:, 2] 

    data3 = np.genfromtxt(file3, delimiter=',', skip_header=1)
    x3 = data3[:, 1] 
    y3 = data3[:, 2] 

    data4 = np.genfromtxt(file4, delimiter=',', skip_header=1)
    x4 = data4[:, 1] 
    y4 = data4[:, 2] 
    
    
    # 플롯 생성
    plt.figure(figsize=(20, 15))  # 그래프 크기 설정
    
    plt.plot(x1, y1, 'o-', label='Waypoint1', markersize=3)  # 투명도 설정하여 겹치는 부분을 줄임
    plt.plot(x2, y2, 'o-', label='Waypoint2', markersize=3)
    plt.plot(x3, y3, 'o-', label='Waypoint3', markersize=3)
    plt.plot(x4, y4, 'o-', label='Waypoint4', markersize=3)
    
    plt.title('Waypoint Visualization')  # 그래프 제목
    plt.xlabel('X Position')  # x축 레이블
    plt.ylabel('Y Position')  # y축 레이블
    plt.legend()
    plt.grid(True)  # 그리드 표시
    plt.show()  # 그래프 표시

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: python3 viewWaypoints.py file1.csv file2.csv file3.csv file4.csv")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]
    file4 = sys.argv[4]
    
    main(file1, file2, file3, file4)
