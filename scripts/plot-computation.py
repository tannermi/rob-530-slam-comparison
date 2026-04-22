import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Setup visual style
colors = {
    'FAST-LIO2': '#c44e52',          # Red
    'DropD-SLAM': '#55a868',         # Green
    'DropD-SLAM_dynamic': '#ccb974', # Gold/Yellow
    'ORB-SLAM3': '#4c72b0'           # Blue
}

def parse_execmean(filepath):
    """Extracts the timing metrics from an ExecMean.txt file."""
    metrics = {
        'extraction': 0.0,
        'tracking_total': 0.0,
        'mapping_total': 0.0,
        'ai_inference': 0.0
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract just the float
            ext_match = re.search(r'ORB Extraction:\s*([\d\.]+)', content)
            if ext_match:
                metrics['extraction'] = float(ext_match.group(1))
                
            trk_match = re.search(r'Total Tracking:\s*([\d\.]+)', content)
            if trk_match:
                metrics['tracking_total'] = float(trk_match.group(1))
                
            map_match = re.search(r'Total Local Mapping:\s*([\d\.]+)', content)
            if map_match:
                metrics['mapping_total'] = float(map_match.group(1))
                
            ai_match = re.search(r'mean depth inference time:\s*([\d\.]+)\s*ms', content)
            if ai_match:
                metrics['ai_inference'] = float(ai_match.group(1))
                
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return metrics

def generate_computation_plots():
    # store[dataset][algo][metric] = [list of values]
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Look for ExecMean.txt
    search_patterns = [
        os.path.join('..', 'results', '*', '*', '*', 'ExecMean.txt')
    ]
    
    filepaths = []
    for pattern in search_patterns:
        filepaths.extend(glob.glob(pattern))

    if not filepaths:
        print("No ExecMean.txt files found! Check your directory structure.")
        return

    print(f"Found {len(filepaths)} ExecMean files. Processing...")

    for path in filepaths:
        parts = path.split(os.sep)
        algo = parts[-2]
        dataset = parts[-4]
        
        metrics = parse_execmean(path)

        # Only add if we successfully parsed tracking data
        if metrics['tracking_total'] > 0:
            store[dataset][algo]['extraction'].append(metrics['extraction'])
            # Tracking Math = Total Tracking - Extraction
            math_time = max(0, metrics['tracking_total'] - metrics['extraction'])
            store[dataset][algo]['tracking_math'].append(math_time)
            store[dataset][algo]['mapping_total'].append(metrics['mapping_total'])
            store[dataset][algo]['ai_inference'].append(metrics['ai_inference'])

    # Generate plots per dataset
    for dataset, algos_data in store.items():
        print(f"\nGenerating computation plots for {dataset}...")
        algorithms = list(algos_data.keys())
        
        # Calculate averages
        avg_math = [np.mean(algos_data[a]['tracking_math']) for a in algorithms]
        avg_ext = [np.mean(algos_data[a]['extraction']) for a in algorithms]
        avg_ai = [np.mean(algos_data[a]['ai_inference']) for a in algorithms]
        avg_map = [np.mean(algos_data[a]['mapping_total']) for a in algorithms]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(f'{dataset}: System Computational Profiling', fontsize=16, fontweight='bold', y=1.02)

        # --- PLOT 1: Synchronous Tracking Pipeline ---
        ax1.set_title('Per-Frame Tracking Latency (Synchronous)', fontsize=13, pad=10)
        ax1.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Bottom layer: Tracking Optimization
        p1 = ax1.bar(algorithms, avg_math, color='#4c72b0', edgecolor='black', label='Tracking Optimization')
        # Middle layer: Feature Extraction
        p2 = ax1.bar(algorithms, avg_ext, bottom=avg_math, color='#55a868', edgecolor='black', label='Feature Extraction')
        # Top layer: AI Inference
        bottom_for_ai = np.add(avg_math, avg_ext)
        p3 = ax1.bar(algorithms, avg_ai, bottom=bottom_for_ai, color='#ccb974', edgecolor='black', label='Depth Inference')

        # Add 30 FPS Threshold
        ax1.axhline(y=33.3, color='red', linestyle=':', linewidth=2, label='Real-Time Threshold (33.3 ms)')
        ax1.legend(loc='upper left')

        # Add total text
        for i, algo in enumerate(algorithms):
            total_time = avg_math[i] + avg_ext[i] + avg_ai[i]
            fps = 1000.0 / total_time if total_time > 0 else 0
            ax1.text(i, total_time + max(avg_ai)*0.05 + 2, f'{total_time:.1f} ms\n(~{fps:.1f} FPS)', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # --- PLOT 2: Asynchronous Mapping Thread ---
        ax2.set_title('Local Mapping (Asynchronous)', fontsize=13, pad=10)
        ax2.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        map_colors = [colors.get(a, '#8c8c8c') for a in algorithms]
        bars2 = ax2.bar(algorithms, avg_map, color=map_colors, edgecolor='black', width=0.6)

        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f} ms', ha='center', va='bottom', fontweight='bold')

        # Rotate x-labels if algorithm names are long
        ax1.tick_params(axis='x', rotation=15)
        ax2.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        filename = f'{dataset}_computation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print("\nSuccess! All computation plots have been saved dynamically.")

if __name__ == "__main__":
    generate_computation_plots()
