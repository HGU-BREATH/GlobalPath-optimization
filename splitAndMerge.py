import pandas as pd
import numpy as np
import os
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def load_inflection_points(file_path):
    """Load inflection points from a CSV file."""
    return pd.read_csv(file_path)

def load_trajectory(file_path):
    """Load a trajectory from a CSV file."""
    return pd.read_csv(file_path)

def load_waypoints(waypoint_dir, source_index):
    filename = f"waypoint{source_index+1}.csv"
    file_path = os.path.join(waypoint_dir, filename)
    return pd.read_csv(file_path)

def find_closest_points(inflection_points, trajectory):
    """Find the closest points in the trajectory to each inflection point."""
    tree = KDTree(trajectory[['x', 'y']].values)
    distances, indices = tree.query(inflection_points[['x', 'y']].values)
    return trajectory.iloc[indices]

def split_trajectory_by_inflection_points(trajectory, inflection_points):
    """Split the trajectory into segments based on the inflection points."""
    inflection_indices = find_closest_points(inflection_points, trajectory).index
    segments = []
    # start_idx = 0
    # print("inflection points: ", inflection_indices)
    for i, idx in enumerate(inflection_indices):
        
        #print("inflection point:", inflection_indices[i+1])
        if(i == len(inflection_indices)-1):
            #print("last index!", idx, inflection_indices[0], "\n---------------------------")
            segments.append(pd.concat([trajectory.iloc[idx:], trajectory.iloc[:inflection_indices[0]+1]]))
        elif(inflection_indices[i+1] < idx):
            #print("elif")
            segments.append(pd.concat([trajectory.iloc[idx:], trajectory.iloc[:inflection_indices[i+1]+1]]))
        else:
            #print(idx, inflection_indices[i+1])
            segments.append(trajectory.iloc[idx:inflection_indices[i+1]+1])
        
    # print(segments)
    return segments


def calculate_intervals(segments):
    """Calculate the interval times for each segment."""
    intervals = []
    for segment in segments:
        if segment['t'].iloc[-1] < segment['t'].iloc[0]:
            # Handle wrap-around case
            interval_time = (segment['t'].iloc[-1] + segment['t'].max()) - segment['t'].iloc[0]
        else:
            interval_time = segment['t'].iloc[-1] - segment['t'].iloc[0]
        intervals.append(interval_time.round(4))
    return intervals

def save_segments(segments, base_name, output_dir):
    """Save each segment to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    for i, segment in enumerate(segments):
        segment_file = os.path.join(output_dir, f"{base_name}_segment_{i+1}.csv")
        segment.to_csv(segment_file, index=False)

def find_fastest_intervals(all_intervals, all_segments):
    """Find the fastest interval for each segment."""
    fastest_intervals = []
    fastest_segments = []
    fastest_sources = []

    num_intervals = len(all_intervals[0])

    for i in range(num_intervals):
        min_time = float('inf')
        min_index = -1

        for j in range(len(all_intervals)):
            if all_intervals[j][i] < min_time:
                min_time = all_intervals[j][i]
                min_index = j
        
        fastest_intervals.append(min_time)
        fastest_segments.append(all_segments[min_index][i])
        fastest_sources.append(min_index)
    
    return fastest_intervals, fastest_segments, fastest_sources

def find_closest_waypoint_points(inflection_points, waypoint):
    tree = KDTree(waypoint[['x', 'y']].values)
    distances, indices = tree.query(inflection_points[['x', 'y']].values)
    return waypoint.iloc[indices]

def create_optimized_waypoints(fastest_segments, fastest_sources, waypoint_dir, inflection_points):
    def smooth_transition(segment1, segment2, num_points=6):
        """Smoothly transition between the end of segment1 and the start of segment2."""
        end_segment1 = segment1.iloc[-num_points:].copy()
        start_segment2 = segment2.iloc[:num_points].copy()

        # Create interpolators for position and speed
        t1 = np.linspace(0, 1, num_points)
        t2 = np.linspace(1, 2, num_points)
        t_combined = np.concatenate((t1, t2))

        x_combined = np.concatenate((end_segment1['x'].values, start_segment2['x'].values))
        y_combined = np.concatenate((end_segment1['y'].values, start_segment2['y'].values))
        v_combined = np.concatenate((end_segment1['v'].values, start_segment2['v'].values))

        # Remove duplicates
        t_combined, unique_indices = np.unique(t_combined, return_index=True)
        x_combined = x_combined[unique_indices]
        y_combined = y_combined[unique_indices]
        v_combined = v_combined[unique_indices]

        interp_x = interp1d(t_combined, x_combined, kind='cubic')
        interp_y = interp1d(t_combined, y_combined, kind='cubic')
        interp_v = interp1d(t_combined, v_combined, kind='cubic')

        t_smooth = np.linspace(0, 2, 2*num_points)
        x_smooth = interp_x(t_smooth)
        y_smooth = interp_y(t_smooth)
        v_smooth = interp_v(t_smooth)

        # Create smooth transition segments
        smooth_end_segment1 = pd.DataFrame({
            'x': x_smooth[:num_points],
            'y': y_smooth[:num_points],
            'v': v_smooth[:num_points]
        })

        smooth_start_segment2 = pd.DataFrame({
            'x': x_smooth[num_points:],
            'y': y_smooth[num_points:],
            'v': v_smooth[num_points:]
        })

        segment1.loc[segment1.index[-num_points:], ['x', 'y', 'v']] = smooth_end_segment1.values
        segment2.loc[segment2.index[:num_points], ['x', 'y', 'v']] = smooth_start_segment2.values

        return segment1, segment2

    optimized_waypoints = []
    
    for i, source_index in enumerate(fastest_sources):
        # Load the corresponding waypoint
        waypoint = load_waypoints(waypoint_dir, source_index)
        
        # Find the closest points in the waypoint to the inflection points
        closest_points = find_closest_waypoint_points(inflection_points, waypoint)
        
        # Determine the segment in the waypoint corresponding to the optimized segment
        start_idx = closest_points.index[i]
        end_idx = closest_points.index[(i+1) % len(closest_points)]
        
        if start_idx < end_idx:
            segment = waypoint.iloc[start_idx:end_idx+1]
        else:
            segment = pd.concat([waypoint.iloc[start_idx:], waypoint.iloc[:end_idx+1]])
        
        optimized_waypoints.append(segment)

    # Adjust segments to ensure smooth transitions
    for i in range(len(optimized_waypoints) - 1):
        optimized_waypoints[i], optimized_waypoints[i+1] = smooth_transition(optimized_waypoints[i], optimized_waypoints[i+1])
    
    optimized_waypoints[-1], optimized_waypoints[0] = smooth_transition(optimized_waypoints[-1], optimized_waypoints[0])

    optimized_waypoints_df = pd.concat(optimized_waypoints).reset_index(drop=True)
    optimized_waypoints_df['x'] = optimized_waypoints_df['x'].round(5)
    optimized_waypoints_df['y'] = optimized_waypoints_df['y'].round(5)
    optimized_waypoints_df['v'] = optimized_waypoints_df['v'].round(5)
    
    first_waypoint = optimized_waypoints_df.iloc[0:1].copy()
    optimized_waypoints_df = pd.concat([optimized_waypoints_df, first_waypoint], ignore_index=True)

    return optimized_waypoints_df

def main(inflection_points_file, trajectory_dir, waypoint_dir):
    input_dir = './src/inflectionPoints/'
    inflection_points_file = os.path.join(input_dir, inflection_points_file)

    # 설정한 출력 디렉토리
    output_dir = './src/output_segments/'  # 여기에 출력 디렉토리를 설정

    # Load inflection points
    inflection_points = load_inflection_points(inflection_points_file)
    
    # 결과를 저장할 데이터 구조
    all_intervals = []
    all_segments = []
    
    # 지정된 디렉토리 내의 모든 CSV 파일 처리 (정렬된 순서로)
    trajectory_files = sorted(f for f in os.listdir(trajectory_dir) if f.endswith('.csv'))
    
    for filename in trajectory_files:
        trajectory_file = os.path.join(trajectory_dir, filename)
        
        # Load trajectory
        trajectory = load_trajectory(trajectory_file)
        
        # Split trajectory into segments
        segments = split_trajectory_by_inflection_points(trajectory, inflection_points)
        '''
        for i, segment in enumerate(segments):
            print(f"segment{i}:\n--------------",segment)
        '''
        # Calculate interval times
        intervals = calculate_intervals(segments)
        all_intervals.append(intervals)
        all_segments.append(segments)
        
        # Save segments
        # base_name = os.path.splitext(filename)[0]
        # save_segments(segments, base_name, output_dir)
    
    # Find the fastest intervals and their sources
    fastest_intervals, fastest_segments, fastest_sources = find_fastest_intervals(all_intervals, all_segments)

    # 결과 출력 (테스트 목적)
    print("Fastest Intervals and Their Sources:\n-----------------------------------")
    for i, interval in enumerate(fastest_intervals):
        print(f"  Interval {i+1}: {interval}, Source: Trajectory {fastest_sources[i]+1}")
    
    optWaypointDf = create_optimized_waypoints(fastest_segments, fastest_sources, waypoint_dir, inflection_points)

    optWaypointFile = './src/optWaypoints/optWaypoint.csv'
    os.makedirs(os.path.dirname(optWaypointFile), exist_ok=True)
    optWaypointDf.to_csv(optWaypointFile, index=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split trajectories by inflection points and calculate interval times.')
    parser.add_argument('inflection_points_file', type=str, help='Path to the inflection points CSV file.')
    parser.add_argument('trajectory_dir', type=str, help='Directory containing trajectory CSV files to process.')
    parser.add_argument('waypoints_dir', type=str, help='Directory containing waypoints CSV files to process')
    args = parser.parse_args()
    
    main(args.inflection_points_file, args.trajectory_dir, args.waypoints_dir)