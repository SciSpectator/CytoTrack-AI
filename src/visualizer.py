"""
Cell Tracker - Publication Quality Visualization
=================================================
Fixed: No overlapping legends, single color for single type
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from typing import Dict, List, Optional
import pandas as pd
from scipy import stats

# Plotly for interactive
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme for cell types
TYPE_COLORS = {
    'Cell': '#1f77b4',
    'default': '#1f77b4',
    'Type_A': '#1f77b4',
    'Type_B': '#ff7f0e',
    'Type_C': '#2ca02c',
    'Type_D': '#d62728',
    'Healthy': '#2ca02c',
    'Cancer': '#d62728',
    'Treated': '#1f77b4',
    'Control': '#ff7f0e',
}

SINGLE_COLOR = '#1f77b4'  # Blue for single cell type


class TrajectoryVisualizer:
    """Publication-quality cell migration plots."""
    
    def __init__(self, pixel_size_x: float = 1.0, pixel_size_y: float = 1.0,
                 time_per_frame: float = 60.0):
        self.px_x = pixel_size_x
        self.px_y = pixel_size_y
        self.time_per_frame = time_per_frame
    
    def _get_type_color(self, cell_type: str) -> str:
        if cell_type in TYPE_COLORS:
            return TYPE_COLORS[cell_type]
        # Generate consistent color
        hash_val = hash(cell_type) % 8
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        return colors[hash_val]
    
    def _extract_trajectories(self, tracks: Dict) -> Dict:
        """Extract normalized trajectories from tracks."""
        trajectories = {}
        
        for tid, track in tracks.items():
            boxes = track.get('boxes', [])
            if len(boxes) < 2:
                continue
            
            x = np.array([(b[0] + b[2]/2) * self.px_x for b in boxes])
            y = np.array([(b[1] + b[3]/2) * self.px_y for b in boxes])
            
            trajectories[tid] = {
                'x': x,
                'y': y,
                'x_norm': x - x[0],
                'y_norm': y - y[0],
                'cell_type': track.get('cell_type', 'Cell'),
                'frames': len(boxes)
            }
        
        return trajectories
    
    def _get_unique_types(self, trajectories: Dict) -> List[str]:
        """Get unique cell types."""
        return list(set(t['cell_type'] for t in trajectories.values()))
    
    # ==================== CIRCULAR TRAJECTORY PLOT ====================
    
    def plot_circular_trajectories(self, tracks: Dict, output_path: str,
                                   title: str = "Cell Migration Trajectories") -> str:
        """Circular trajectory plot - single color if one type, colors if multiple."""
        trajectories = self._extract_trajectories(tracks)
        
        if not trajectories:
            return None
        
        types = self._get_unique_types(trajectories)
        is_single_type = len(types) <= 1
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Find max displacement
        max_disp = 0
        for traj in trajectories.values():
            max_disp = max(max_disp, np.max(np.abs(traj['x_norm'])), 
                         np.max(np.abs(traj['y_norm'])))
        max_disp = max(max_disp * 1.15, 10)
        
        # Draw concentric circles
        for r in np.linspace(max_disp/4, max_disp, 4):
            circle = Circle((0, 0), r, fill=False, color='#cccccc', 
                           linestyle='--', linewidth=0.5)
            ax.add_patch(circle)
            # Label only on one side
            ax.text(r * 0.71, r * 0.71 + 2, f'{r:.0f}', fontsize=8, 
                   color='#888888', ha='center')
        
        # Axes
        ax.axhline(y=0, color='#cccccc', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='#cccccc', linestyle='-', linewidth=0.5)
        
        # Plot trajectories
        if is_single_type:
            # SINGLE TYPE: All same color
            for traj in trajectories.values():
                ax.plot(traj['x_norm'], traj['y_norm'], color=SINGLE_COLOR, 
                       linewidth=0.8, alpha=0.6)
                ax.scatter(traj['x_norm'][-1], traj['y_norm'][-1], 
                          color=SINGLE_COLOR, s=15, zorder=5, alpha=0.7)
            
            type_name = types[0] if types else 'Cell'
            n_cells = len(trajectories)
            ax.text(0.02, 0.98, f'{type_name} (n={n_cells})', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', color=SINGLE_COLOR,
                   fontweight='bold')
        else:
            # MULTIPLE TYPES: Different colors
            type_counts = {}
            for traj in trajectories.values():
                ct = traj['cell_type']
                type_counts[ct] = type_counts.get(ct, 0) + 1
                color = self._get_type_color(ct)
                ax.plot(traj['x_norm'], traj['y_norm'], color=color, 
                       linewidth=0.8, alpha=0.5)
                ax.scatter(traj['x_norm'][-1], traj['y_norm'][-1], 
                          color=color, s=15, zorder=5, alpha=0.7)
            
            # Legend outside plot
            handles = [mpatches.Patch(color=self._get_type_color(ct), 
                                      label=f'{ct} (n={type_counts[ct]})') 
                      for ct in sorted(type_counts.keys())]
            ax.legend(handles=handles, loc='upper left', framealpha=0.9,
                     bbox_to_anchor=(0.02, 0.98), fontsize=9)
        
        # Center point
        ax.scatter(0, 0, color='black', s=60, marker='o', zorder=10)
        
        ax.set_xlim(-max_disp, max_disp)
        ax.set_ylim(-max_disp, max_disp)
        ax.set_aspect('equal')
        ax.set_xlabel('X displacement (µm)')
        ax.set_ylabel('Y displacement (µm)')
        ax.set_title(title, pad=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_separate_by_type(self, tracks: Dict, output_dir: str) -> List[str]:
        """Generate separate circular plot for each cell type."""
        trajectories = self._extract_trajectories(tracks)
        types = self._get_unique_types(trajectories)
        
        if len(types) <= 1:
            return []  # No need for separate plots
        
        paths = []
        for cell_type in types:
            type_trajs = {k: v for k, v in trajectories.items() 
                        if v['cell_type'] == cell_type}
            
            if len(type_trajs) < 2:
                continue
            
            fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
            ax.set_facecolor('white')
            
            max_disp = 0
            for traj in type_trajs.values():
                max_disp = max(max_disp, np.max(np.abs(traj['x_norm'])), 
                             np.max(np.abs(traj['y_norm'])))
            max_disp = max(max_disp * 1.15, 10)
            
            for r in np.linspace(max_disp/4, max_disp, 4):
                circle = Circle((0, 0), r, fill=False, color='#cccccc', 
                               linestyle='--', linewidth=0.5)
                ax.add_patch(circle)
            
            ax.axhline(y=0, color='#cccccc', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='#cccccc', linestyle='-', linewidth=0.5)
            
            color = self._get_type_color(cell_type)
            for traj in type_trajs.values():
                ax.plot(traj['x_norm'], traj['y_norm'], color=color, 
                       linewidth=0.8, alpha=0.6)
                ax.scatter(traj['x_norm'][-1], traj['y_norm'][-1], 
                          color=color, s=15, zorder=5)
            
            ax.scatter(0, 0, color='black', s=60, marker='o', zorder=10)
            
            ax.set_xlim(-max_disp, max_disp)
            ax.set_ylim(-max_disp, max_disp)
            ax.set_aspect('equal')
            ax.set_xlabel('X displacement (µm)')
            ax.set_ylabel('Y displacement (µm)')
            ax.set_title(f'{cell_type} (n={len(type_trajs)})', pad=10)
            
            safe_name = cell_type.replace(' ', '_').replace('/', '_')
            path = os.path.join(output_dir, f'trajectories_{safe_name}.png')
            plt.tight_layout()
            plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
            plt.close()
            paths.append(path)
        
        return paths
    
    # ==================== VELOCITY PLOTS ====================
    
    def plot_velocity_histogram(self, summary_df: pd.DataFrame, output_path: str) -> str:
        """Velocity distribution histogram."""
        if 'Avg_Velocity_um_min' not in summary_df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white')
        
        velocities = summary_df['Avg_Velocity_um_min'].dropna()
        
        ax.hist(velocities, bins=20, color=SINGLE_COLOR, edgecolor='white', 
               alpha=0.7, density=True)
        
        # KDE
        if len(velocities) > 5:
            kde_x = np.linspace(velocities.min(), velocities.max(), 100)
            kde = stats.gaussian_kde(velocities)
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
        
        mean_v = velocities.mean()
        std_v = velocities.std()
        ax.axvline(mean_v, color='red', linestyle='--', linewidth=1.5)
        
        # Stats box - positioned to not overlap
        stats_text = f'n = {len(velocities)}\nMean = {mean_v:.2f} ± {std_v:.2f} µm/min'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='white', 
                                     edgecolor='gray', alpha=0.9))
        
        ax.set_xlabel('Average Velocity (µm/min)')
        ax.set_ylabel('Density')
        ax.set_title('Velocity Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_velocity_boxplot(self, summary_df: pd.DataFrame, output_path: str) -> str:
        """Velocity box plot comparing cell types."""
        if 'Cell_Type' not in summary_df.columns or 'Avg_Velocity_um_min' not in summary_df.columns:
            return None
        
        types = summary_df['Cell_Type'].unique()
        if len(types) < 2:
            return None  # No comparison needed
        
        fig, ax = plt.subplots(figsize=(max(6, len(types)*1.5), 5), facecolor='white')
        ax.set_facecolor('white')
        
        data = [summary_df[summary_df['Cell_Type'] == t]['Avg_Velocity_um_min'].dropna() 
                for t in types]
        colors = [self._get_type_color(t) for t in types]
        
        bp = ax.boxplot(data, patch_artist=True, labels=types)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Points
        for i, (d, color) in enumerate(zip(data, colors)):
            x = np.random.normal(i+1, 0.04, size=len(d))
            ax.scatter(x, d, color=color, alpha=0.5, s=20, edgecolors='white', linewidths=0.5)
        
        # T-test for 2 groups
        if len(types) == 2 and len(data[0]) > 2 and len(data[1]) > 2:
            _, p_val = stats.ttest_ind(data[0], data[1])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            
            y_max = max(d.max() for d in data)
            ax.plot([1, 1, 2, 2], [y_max*1.05, y_max*1.1, y_max*1.1, y_max*1.05], 'k-', linewidth=1)
            ax.text(1.5, y_max*1.12, f'{sig} (p={p_val:.3f})', ha='center', fontsize=9)
        
        ax.set_ylabel('Average Velocity (µm/min)')
        ax.set_title('Velocity Comparison')
        
        # Sample sizes below x-axis
        for i, (t, d) in enumerate(zip(types, data)):
            ax.text(i+1, ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                   f'n={len(d)}', ha='center', va='top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # ==================== DISPLACEMENT ====================
    
    def plot_displacement_distance(self, summary_df: pd.DataFrame, output_path: str) -> str:
        """Displacement vs Total Distance scatter."""
        if 'Total_Distance_um' not in summary_df.columns or 'Displacement_um' not in summary_df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')
        
        has_types = 'Cell_Type' in summary_df.columns and len(summary_df['Cell_Type'].unique()) > 1
        
        if has_types:
            for ct in summary_df['Cell_Type'].unique():
                mask = summary_df['Cell_Type'] == ct
                color = self._get_type_color(ct)
                n = mask.sum()
                ax.scatter(summary_df.loc[mask, 'Total_Distance_um'],
                          summary_df.loc[mask, 'Displacement_um'],
                          c=color, label=f'{ct} (n={n})', s=40, alpha=0.7, 
                          edgecolors='white', linewidths=0.5)
            ax.legend(loc='upper left', fontsize=9)
        else:
            ax.scatter(summary_df['Total_Distance_um'], summary_df['Displacement_um'],
                      c=SINGLE_COLOR, s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
        
        # CDE=1 line
        max_val = max(summary_df['Total_Distance_um'].max(), summary_df['Displacement_um'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='CDE = 1')
        
        ax.set_xlabel('Total Distance (µm)')
        ax.set_ylabel('Net Displacement (µm)')
        ax.set_title('Displacement vs Distance')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # ==================== DIRECTIONALITY (CDE) ====================
    
    def plot_directionality(self, summary_df: pd.DataFrame, output_path: str) -> str:
        """CDE distribution plot."""
        if 'CDE' not in summary_df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white')
        
        has_types = 'Cell_Type' in summary_df.columns and len(summary_df['Cell_Type'].unique()) > 1
        
        if has_types:
            for ct in summary_df['Cell_Type'].unique():
                mask = summary_df['Cell_Type'] == ct
                data = summary_df.loc[mask, 'CDE'].dropna()
                color = self._get_type_color(ct)
                ax.hist(data, bins=15, color=color, alpha=0.5, 
                       label=f'{ct} (n={len(data)})', edgecolor='white', density=True)
        else:
            cde = summary_df['CDE'].dropna()
            ax.hist(cde, bins=20, color=SINGLE_COLOR, alpha=0.7, edgecolor='white', density=True)
        
        mean_cde = summary_df['CDE'].mean()
        ax.axvline(mean_cde, color='red', linestyle='--', linewidth=1.5)
        ax.text(mean_cde + 0.02, ax.get_ylim()[1]*0.9, f'Mean: {mean_cde:.3f}', 
               fontsize=9, color='red')
        
        ax.set_xlabel('Confinement Ratio (CDE)')
        ax.set_ylabel('Density')
        ax.set_title('Directional Persistence')
        ax.set_xlim(0, 1)
        
        if has_types:
            ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # ==================== MSD ====================
    
    def plot_msd(self, tracks: Dict, output_path: str) -> str:
        """Mean Squared Displacement plot."""
        trajectories = self._extract_trajectories(tracks)
        
        if not trajectories:
            return None
        
        types = self._get_unique_types(trajectories)
        is_single_type = len(types) <= 1
        
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white')
        
        # Group by type
        type_data = {}
        for tid, traj in trajectories.items():
            ct = traj['cell_type']
            if ct not in type_data:
                type_data[ct] = []
            
            x, y = traj['x_norm'], traj['y_norm']
            n = len(x)
            max_lag = min(n - 1, 30)
            
            msd = []
            for lag in range(1, max_lag + 1):
                dx = x[lag:] - x[:-lag]
                dy = y[lag:] - y[:-lag]
                msd.append(np.mean(dx**2 + dy**2))
            
            type_data[ct].append(msd)
        
        # Plot
        for ct, msds in type_data.items():
            color = SINGLE_COLOR if is_single_type else self._get_type_color(ct)
            
            # Individual tracks (light)
            for msd in msds:
                lags = np.arange(1, len(msd) + 1) * self.time_per_frame / 60
                ax.plot(lags, msd, color=color, alpha=0.15, linewidth=0.5)
            
            # Mean MSD
            max_len = max(len(m) for m in msds)
            mean_msd = np.zeros(max_len)
            count = np.zeros(max_len)
            for msd in msds:
                mean_msd[:len(msd)] += msd
                count[:len(msd)] += 1
            mean_msd = mean_msd / np.maximum(count, 1)
            
            lags = np.arange(1, max_len + 1) * self.time_per_frame / 60
            label = f'{ct} (n={len(msds)})' if not is_single_type else f'n={len(msds)}'
            ax.plot(lags, mean_msd, color=color, linewidth=2.5, label=label)
        
        ax.set_xlabel('Time lag (min)')
        ax.set_ylabel('MSD (µm²)')
        ax.set_title('Mean Squared Displacement')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # ==================== ROSE PLOT ====================
    
    def plot_rose(self, tracks: Dict, output_path: str) -> str:
        """Angular distribution rose plot."""
        trajectories = self._extract_trajectories(tracks)
        
        if not trajectories:
            return None
        
        types = self._get_unique_types(trajectories)
        n_types = len(types)
        
        # Collect angles
        type_angles = {ct: [] for ct in types}
        for traj in trajectories.values():
            dx = traj['x_norm'][-1]
            dy = traj['y_norm'][-1]
            if np.sqrt(dx**2 + dy**2) > 1:  # Minimum displacement threshold
                angle = np.arctan2(dy, dx)
                type_angles[traj['cell_type']].append(angle)
        
        # Remove types with no angles
        type_angles = {k: v for k, v in type_angles.items() if len(v) > 0}
        n_types = len(type_angles)
        
        if n_types == 0:
            return None
        
        if n_types == 1:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'},
                                  facecolor='white')
            axes = [ax]
        else:
            cols = min(n_types, 3)
            rows = (n_types + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows),
                                    subplot_kw={'projection': 'polar'},
                                    facecolor='white')
            axes = axes.flatten() if n_types > 1 else [axes]
        
        for ax, (ct, angles) in zip(axes, type_angles.items()):
            angles = np.array(angles)
            color = SINGLE_COLOR if n_types == 1 else self._get_type_color(ct)
            
            bins = np.linspace(-np.pi, np.pi, 17)
            hist, _ = np.histogram(angles, bins=bins)
            width = 2 * np.pi / 16
            centers = (bins[:-1] + bins[1:]) / 2
            
            ax.bar(centers, hist, width=width, color=color, alpha=0.7, edgecolor='white')
            ax.set_title(f'{ct} (n={len(angles)})', pad=10)
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(-1)
        
        # Hide unused axes
        for ax in axes[n_types:]:
            ax.set_visible(False)
        
        plt.suptitle('Angular Distribution of Migration', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # ==================== SUMMARY COMPARISON ====================
    
    def plot_summary_comparison(self, summary_df: pd.DataFrame, output_path: str) -> str:
        """Multi-panel comparison of metrics."""
        if 'Cell_Type' not in summary_df.columns:
            return None
        
        types = summary_df['Cell_Type'].unique()
        if len(types) < 2:
            return None
        
        metrics = ['Avg_Velocity_um_min', 'Displacement_um', 'Total_Distance_um', 'CDE']
        labels = ['Velocity\n(µm/min)', 'Displacement\n(µm)', 'Total Distance\n(µm)', 'CDE']
        
        available = [(m, l) for m, l in zip(metrics, labels) if m in summary_df.columns]
        n = len(available)
        
        if n == 0:
            return None
        
        fig, axes = plt.subplots(1, n, figsize=(3.5*n, 5), facecolor='white')
        if n == 1:
            axes = [axes]
        
        for ax, (metric, label) in zip(axes, available):
            ax.set_facecolor('white')
            
            data = [summary_df[summary_df['Cell_Type'] == t][metric].dropna() for t in types]
            colors = [self._get_type_color(t) for t in types]
            
            bp = ax.boxplot(data, patch_artist=True, labels=types)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            for i, (d, color) in enumerate(zip(data, colors)):
                x = np.random.normal(i+1, 0.04, size=len(d))
                ax.scatter(x, d, color=color, alpha=0.4, s=15, edgecolors='white', linewidths=0.3)
            
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Cell Migration Parameters', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        
        return output_path
    
    # ==================== INTERACTIVE ====================
    
    def plot_interactive(self, tracks: Dict, output_path: str) -> str:
        """Interactive Plotly plot."""
        if not HAS_PLOTLY:
            return None
        
        trajectories = self._extract_trajectories(tracks)
        if not trajectories:
            return None
        
        types = self._get_unique_types(trajectories)
        is_single = len(types) <= 1
        
        fig = go.Figure()
        
        for tid, traj in trajectories.items():
            color = SINGLE_COLOR if is_single else self._get_type_color(traj['cell_type'])
            
            fig.add_trace(go.Scatter(
                x=traj['x_norm'].tolist(), y=traj['y_norm'].tolist(),
                mode='lines',
                name=f"ID {tid}" if is_single else f"ID {tid} ({traj['cell_type']})",
                line=dict(width=1, color=color),
                opacity=0.6,
                hovertemplate=f'ID {tid}<br>X: %{{x:.1f}} µm<br>Y: %{{y:.1f}} µm'
            ))
        
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=12, color='black'),
            name='Start',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Interactive Cell Trajectories',
            xaxis_title='X displacement (µm)',
            yaxis_title='Y displacement (µm)',
            template='plotly_white',
            hovermode='closest',
            xaxis=dict(scaleanchor="y"),
            showlegend=not is_single or len(trajectories) < 20
        )
        
        fig.write_html(output_path)
        return output_path
    
    # ==================== GENERATE ALL ====================
    
    def generate_all_plots(self, tracks: Dict, detailed_df: pd.DataFrame,
                          summary_df: pd.DataFrame, output_dir: str) -> List[str]:
        """Generate all publication plots."""
        plots = []
        
        print("📊 Generating publication-quality plots...")
        
        # 1. Circular trajectories
        p = os.path.join(output_dir, 'plot_trajectories.png')
        if self.plot_circular_trajectories(tracks, p):
            plots.append(p)
            print("  ✓ Circular trajectories")
        
        # 2. Separate by type (only if multiple)
        sep = self.plot_separate_by_type(tracks, output_dir)
        plots.extend(sep)
        if sep:
            print(f"  ✓ Separate plots ({len(sep)} types)")
        
        # 3. Velocity histogram
        p = os.path.join(output_dir, 'plot_velocity_histogram.png')
        if self.plot_velocity_histogram(summary_df, p):
            plots.append(p)
            print("  ✓ Velocity histogram")
        
        # 4. Velocity comparison (if multiple)
        p = os.path.join(output_dir, 'plot_velocity_comparison.png')
        if self.plot_velocity_boxplot(summary_df, p):
            plots.append(p)
            print("  ✓ Velocity comparison")
        
        # 5. Displacement vs Distance
        p = os.path.join(output_dir, 'plot_displacement_distance.png')
        if self.plot_displacement_distance(summary_df, p):
            plots.append(p)
            print("  ✓ Displacement vs Distance")
        
        # 6. Directionality (CDE)
        p = os.path.join(output_dir, 'plot_directionality.png')
        if self.plot_directionality(summary_df, p):
            plots.append(p)
            print("  ✓ Directionality (CDE)")
        
        # 7. MSD
        p = os.path.join(output_dir, 'plot_msd.png')
        if self.plot_msd(tracks, p):
            plots.append(p)
            print("  ✓ MSD")
        
        # 8. Rose plot
        p = os.path.join(output_dir, 'plot_rose.png')
        if self.plot_rose(tracks, p):
            plots.append(p)
            print("  ✓ Rose plot")
        
        # 9. Summary comparison (if multiple)
        p = os.path.join(output_dir, 'plot_summary.png')
        if self.plot_summary_comparison(summary_df, p):
            plots.append(p)
            print("  ✓ Summary comparison")
        
        # 10. Interactive
        if HAS_PLOTLY:
            p = os.path.join(output_dir, 'plot_interactive.html')
            if self.plot_interactive(tracks, p):
                plots.append(p)
                print("  ✓ Interactive HTML")
        
        print(f"📊 Generated {len(plots)} plots")
        return [p for p in plots if p]
