"""
Cell Tracker - Migration Analysis Module
=========================================
Complete cell migration statistics for publication.
Includes all standard metrics used in cell biology papers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import stats


class MigrationAnalyzer:
    """
    Comprehensive cell migration analysis.
    
    Calculates all standard metrics:
    - Velocity (instantaneous, average, max)
    - Displacement (net distance from start)
    - Total distance traveled
    - Confinement ratio / Directionality (CDE)
    - Persistence
    - Mean Squared Displacement (MSD)
    - Directional autocorrelation
    - Angular metrics
    """
    
    def __init__(self, pixel_size_x: float = 1.0, pixel_size_y: float = 1.0, 
                 time_per_frame: float = 60.0):
        """
        Initialize analyzer.
        
        Args:
            pixel_size_x: Pixel size in X (µm/pixel)
            pixel_size_y: Pixel size in Y (µm/pixel)
            time_per_frame: Time between frames (seconds)
        """
        self.px_x = pixel_size_x
        self.px_y = pixel_size_y
        self.time_per_frame = time_per_frame  # seconds
        self.time_per_frame_min = time_per_frame / 60.0  # minutes

    def analyze(self, tracks: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze all tracks and return detailed and summary dataframes.
        
        Args:
            tracks: Dictionary of track data
            
        Returns:
            Tuple of (detailed_df, summary_df)
        """
        detailed_data = []
        summary_data = []
        
        for tid, track in tracks.items():
            boxes = track['boxes']
            cell_type = track.get('cell_type', 'Cell')
            
            if len(boxes) < 2:
                continue
            
            # Extract centers in µm
            centers = []
            for x, y, w, h in boxes:
                cx = (x + w/2) * self.px_x
                cy = (y + h/2) * self.px_y
                centers.append((cx, cy))
            
            centers = np.array(centers)
            n_frames = len(centers)
            
            # Calculate step-by-step metrics
            step_distances = []
            step_velocities = []
            step_angles = []
            cumulative_distance = 0
            
            for i in range(n_frames):
                x_um, y_um = centers[i]
                
                if i > 0:
                    dx = centers[i, 0] - centers[i-1, 0]
                    dy = centers[i, 1] - centers[i-1, 1]
                    step_dist = np.sqrt(dx**2 + dy**2)
                    step_vel = step_dist / self.time_per_frame_min
                    step_angle = np.arctan2(dy, dx) * 180 / np.pi  # degrees
                    
                    cumulative_distance += step_dist
                    step_distances.append(step_dist)
                    step_velocities.append(step_vel)
                    step_angles.append(step_angle)
                else:
                    step_dist = 0
                    step_vel = 0
                    step_angle = 0
                
                # Displacement from start
                displacement = np.sqrt((x_um - centers[0, 0])**2 + 
                                       (y_um - centers[0, 1])**2)
                
                detailed_data.append({
                    'TrackID': tid,
                    'Cell_Type': cell_type,
                    'Frame': i,
                    'Time_min': i * self.time_per_frame_min,
                    'X_um': x_um,
                    'Y_um': y_um,
                    'X_displacement_um': x_um - centers[0, 0],
                    'Y_displacement_um': y_um - centers[0, 1],
                    'Step_Distance_um': step_dist,
                    'Cumulative_Distance_um': cumulative_distance,
                    'Displacement_um': displacement,
                    'Velocity_um_min': step_vel,
                    'Step_Angle_deg': step_angle
                })
            
            # Calculate summary statistics
            total_distance = cumulative_distance
            
            # Net displacement (start to end)
            final_displacement = np.sqrt(
                (centers[-1, 0] - centers[0, 0])**2 + 
                (centers[-1, 1] - centers[0, 1])**2
            )
            
            # Confinement ratio / Directionality (CDE)
            # CDE = displacement / total_distance
            # CDE = 1 means straight line, CDE → 0 means random/confined
            cde = final_displacement / total_distance if total_distance > 0 else 0
            
            # Velocity statistics
            avg_velocity = np.mean(step_velocities) if step_velocities else 0
            max_velocity = np.max(step_velocities) if step_velocities else 0
            std_velocity = np.std(step_velocities) if step_velocities else 0
            
            # Persistence (directional autocorrelation)
            persistence = self._calculate_persistence(step_angles)
            
            # Mean squared displacement at different time lags
            msd_data = self._calculate_msd(centers)
            
            # Directional bias
            final_angle = np.arctan2(
                centers[-1, 1] - centers[0, 1],
                centers[-1, 0] - centers[0, 0]
            ) * 180 / np.pi
            
            # Angular standard deviation
            if step_angles:
                # Circular standard deviation
                angles_rad = np.array(step_angles) * np.pi / 180
                angular_std = np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * angles_rad)))))
                angular_std_deg = angular_std * 180 / np.pi
            else:
                angular_std_deg = 0
            
            # Track duration
            duration_min = (n_frames - 1) * self.time_per_frame_min
            
            summary_data.append({
                'TrackID': tid,
                'Cell_Type': cell_type,
                'Frames': n_frames,
                'Duration_min': duration_min,
                'Start_X_um': centers[0, 0],
                'Start_Y_um': centers[0, 1],
                'End_X_um': centers[-1, 0],
                'End_Y_um': centers[-1, 1],
                'Total_Distance_um': total_distance,
                'Displacement_um': final_displacement,
                'CDE': cde,
                'Avg_Velocity_um_min': avg_velocity,
                'Max_Velocity_um_min': max_velocity,
                'Std_Velocity_um_min': std_velocity,
                'Persistence': persistence,
                'Final_Angle_deg': final_angle,
                'Angular_Std_deg': angular_std_deg,
                'MSD_10': msd_data.get(10, np.nan),
                'MSD_20': msd_data.get(20, np.nan),
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        summary_df = pd.DataFrame(summary_data)
        
        return detailed_df, summary_df
    
    def _calculate_persistence(self, angles: List[float]) -> float:
        """
        Calculate directional persistence (autocorrelation of direction).
        
        High persistence = cell tends to maintain direction
        Low persistence = random direction changes
        """
        if len(angles) < 3:
            return np.nan
        
        angles_rad = np.array(angles) * np.pi / 180
        
        # Calculate cosine of angle changes
        angle_changes = np.diff(angles_rad)
        
        # Mean cosine of direction changes
        # cos(0) = 1 means same direction, cos(180°) = -1 means reversal
        persistence = np.mean(np.cos(angle_changes))
        
        return persistence
    
    def _calculate_msd(self, centers: np.ndarray) -> Dict[int, float]:
        """
        Calculate Mean Squared Displacement at specific time lags.
        
        MSD(τ) = <(r(t+τ) - r(t))²>
        
        Returns dict with lag (frames) -> MSD (µm²)
        """
        n = len(centers)
        msd_data = {}
        
        for lag in [5, 10, 20, 30, 50]:
            if lag >= n:
                continue
            
            displacements_squared = []
            for i in range(n - lag):
                dx = centers[i + lag, 0] - centers[i, 0]
                dy = centers[i + lag, 1] - centers[i, 1]
                displacements_squared.append(dx**2 + dy**2)
            
            if displacements_squared:
                msd_data[lag] = np.mean(displacements_squared)
        
        return msd_data
    
    def compare_types(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistical comparison between cell types.
        
        Returns dataframe with comparison statistics.
        """
        if 'Cell_Type' not in summary_df.columns:
            return None
        
        types = summary_df['Cell_Type'].unique()
        if len(types) < 2:
            return None
        
        metrics = ['Avg_Velocity_um_min', 'Displacement_um', 'Total_Distance_um', 
                   'CDE', 'Persistence']
        
        comparisons = []
        
        for metric in metrics:
            if metric not in summary_df.columns:
                continue
            
            for i, type1 in enumerate(types):
                for type2 in types[i+1:]:
                    data1 = summary_df[summary_df['Cell_Type'] == type1][metric].dropna()
                    data2 = summary_df[summary_df['Cell_Type'] == type2][metric].dropna()
                    
                    if len(data1) < 3 or len(data2) < 3:
                        continue
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + 
                                         (len(data2)-1)*data2.std()**2) / 
                                        (len(data1) + len(data2) - 2))
                    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    comparisons.append({
                        'Metric': metric,
                        'Type_1': type1,
                        'Type_2': type2,
                        'Mean_1': data1.mean(),
                        'Mean_2': data2.mean(),
                        'Std_1': data1.std(),
                        'Std_2': data2.std(),
                        'N_1': len(data1),
                        'N_2': len(data2),
                        'T_statistic': t_stat,
                        'T_test_p_value': p_value,
                        'Mann_Whitney_p_value': u_pvalue,
                        'Cohens_d': cohens_d,
                        'Significant_005': p_value < 0.05,
                        'Significant_001': p_value < 0.01
                    })
        
        return pd.DataFrame(comparisons) if comparisons else None
    
    def get_type_summary(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics grouped by cell type.
        """
        if 'Cell_Type' not in summary_df.columns:
            return None
        
        metrics = ['Avg_Velocity_um_min', 'Displacement_um', 'Total_Distance_um', 
                   'CDE', 'Persistence', 'Duration_min']
        
        available_metrics = [m for m in metrics if m in summary_df.columns]
        
        # Group by cell type and calculate statistics
        grouped = summary_df.groupby('Cell_Type')[available_metrics].agg(['mean', 'std', 'median', 'count'])
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        return grouped.reset_index()
