#!/usr/bin/env python3
"""
GPU monitoring script for tracking VRAM usage during training.
Run this in a separate terminal while training is running.

Usage:
    python monitor_gpu.py [--interval 5] [--output gpu_usage.csv]
"""

import argparse
import csv
import time
import datetime
import subprocess
import json
import os
import signal
import sys
from pathlib import Path

def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        # Run nvidia-smi and get output in JSON format
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=timestamp,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        gpu_data = []
        
        for i, line in enumerate(lines):
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 8:
                    gpu_data.append({
                        'gpu_id': i,
                        'timestamp': parts[0],
                        'name': parts[1],
                        'memory_used_mb': int(parts[2]),
                        'memory_total_mb': int(parts[3]),
                        'memory_free_mb': int(parts[4]),
                        'gpu_utilization': int(parts[5]),
                        'temperature': int(parts[6]),
                        'power_draw': float(parts[7]) if parts[7] != '[Not Supported]' else 0.0
                    })
        
        return gpu_data
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Error getting GPU info: {e}")
        return []

def format_memory(mb):
    """Format memory in MB to human readable format."""
    if mb >= 1024:
        return f"{mb/1024:.1f}GB"
    else:
        return f"{mb}MB"

def print_gpu_status(gpu_data):
    """Print current GPU status to console."""
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[{current_time}] GPU Status:")
    print("-" * 80)
    
    for gpu in gpu_data:
        memory_used = gpu['memory_used_mb']
        memory_total = gpu['memory_total_mb']
        memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
        
        print(f"GPU {gpu['gpu_id']} ({gpu['name']}):")
        print(f"  VRAM: {format_memory(memory_used)}/{format_memory(memory_total)} ({memory_percent:.1f}%)")
        print(f"  Utilization: {gpu['gpu_utilization']}%")
        print(f"  Temperature: {gpu['temperature']}°C")
        if gpu['power_draw'] > 0:
            print(f"  Power: {gpu['power_draw']:.1f}W")
        
        # VRAM usage bar
        bar_length = 40
        filled = int(bar_length * memory_percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"  VRAM: [{bar}] {memory_percent:.1f}%")

def save_to_csv(gpu_data, csv_file, write_header=False):
    """Save GPU data to CSV file."""
    if not gpu_data:
        return
    
    fieldnames = [
        'datetime', 'gpu_id', 'gpu_name', 'memory_used_mb', 'memory_total_mb', 
        'memory_free_mb', 'memory_used_percent', 'gpu_utilization', 
        'temperature', 'power_draw'
    ]
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        current_time = datetime.datetime.now().isoformat()
        
        for gpu in gpu_data:
            memory_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100 if gpu['memory_total_mb'] > 0 else 0
            
            row = {
                'datetime': current_time,
                'gpu_id': gpu['gpu_id'],
                'gpu_name': gpu['name'],
                'memory_used_mb': gpu['memory_used_mb'],
                'memory_total_mb': gpu['memory_total_mb'],
                'memory_free_mb': gpu['memory_free_mb'],
                'memory_used_percent': round(memory_percent, 2),
                'gpu_utilization': gpu['gpu_utilization'],
                'temperature': gpu['temperature'],
                'power_draw': gpu['power_draw']
            }
            writer.writerow(row)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nMonitoring stopped by user.")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU usage during training')
    parser.add_argument('--interval', '-i', type=int, default=5, 
                       help='Monitoring interval in seconds (default: 5)')
    parser.add_argument('--output', '-o', type=str, default='gpu_usage.csv',
                       help='Output CSV file (default: gpu_usage.csv)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only save to CSV, no console output')
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if nvidia-smi is available
    try:
        subprocess.run(['nvidia-smi', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we need to write header
    write_header = not output_path.exists()
    
    if not args.quiet:
        print(f"GPU Monitoring Started")
        print(f"Interval: {args.interval} seconds")
        print(f"Output: {args.output}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    try:
        while True:
            gpu_data = get_gpu_info()
            
            if gpu_data:
                # Save to CSV
                save_to_csv(gpu_data, args.output, write_header)
                write_header = False
                
                # Print to console (unless quiet mode)
                if not args.quiet:
                    print_gpu_status(gpu_data)
            else:
                if not args.quiet:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No GPU data available")
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main() 