#!/usr/bin/env python3
"""
Kill stuck training processes and free GPU memory.

Usage:
    python kill_training.py
    # or
    ./kill_training.py
"""

import subprocess
import os
import signal
import time

def run_command(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def get_gpu_memory():
    """Get current GPU memory usage."""
    output = run_command("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
    if output and "Error" not in output:
        parts = output.split(',')
        if len(parts) == 2:
            used = int(parts[0].strip())
            total = int(parts[1].strip())
            percent = (used / total) * 100
            return used, total, percent
    return None, None, None

def find_training_processes():
    """Find all Python training processes."""
    cmd = "ps aux | grep -E 'python.*train\\.py|python.*training' | grep -v grep"
    output = run_command(cmd)
    
    if not output:
        return []
    
    processes = []
    for line in output.split('\n'):
        if line:
            parts = line.split()
            if len(parts) >= 2:
                pid = parts[1]
                try:
                    processes.append(int(pid))
                except ValueError:
                    pass
    
    return processes

def find_zombie_processes():
    """Find zombie processes."""
    cmd = "ps aux | awk '$8==\"Z\" {print $2}'"
    output = run_command(cmd)
    
    if not output:
        return []
    
    zombies = []
    for line in output.split('\n'):
        if line.strip():
            try:
                zombies.append(int(line.strip()))
            except ValueError:
                pass
    
    return zombies

def kill_process(pid):
    """Kill a process by PID."""
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False  # Already dead
    except PermissionError:
        print(f"  ⚠️  Permission denied for PID {pid}")
        return False
    except Exception as e:
        print(f"  ⚠️  Failed to kill PID {pid}: {e}")
        return False

def main():
    print("=" * 60)
    print("🔧 Killing Stuck Training Processes")
    print("=" * 60)
    
    # Show initial GPU status
    used, total, percent = get_gpu_memory()
    if used is not None:
        print(f"\n📊 Current GPU Memory: {used} MB / {total} MB ({percent:.1f}% used)")
    
    # Find and kill training processes
    print("\n🔍 Finding Python training processes...")
    pids = find_training_processes()
    
    if not pids:
        print("✅ No training processes found!")
    else:
        print(f"Found {len(pids)} process(es): {pids}")
        print("\n⚠️  Killing processes...")
        
        killed = 0
        for pid in pids:
            if kill_process(pid):
                print(f"  ✓ Killed PID: {pid}")
                killed += 1
            else:
                print(f"  ✗ Could not kill PID: {pid}")
        
        print(f"\nKilled {killed}/{len(pids)} processes")
    
    # Find and kill zombie processes
    print("\n🧟 Checking for zombie processes...")
    zombies = find_zombie_processes()
    
    if zombies:
        print(f"Found {len(zombies)} zombie(s): {zombies}")
        for zpid in zombies:
            kill_process(zpid)
            print(f"  ✓ Killed zombie PID: {zpid}")
    else:
        print("✅ No zombies found!")
    
    # Clear CUDA cache
    print("\n🧹 Clearing CUDA cache...")
    try:
        import torch
        torch.cuda.empty_cache()
        print("  ✓ CUDA cache cleared")
    except ImportError:
        print("  ⚠️  PyTorch not available, skipping cache clear")
    except Exception as e:
        print(f"  ⚠️  Error clearing cache: {e}")
    
    # Wait for processes to fully die
    time.sleep(2)
    
    # Show final GPU status
    used, total, percent = get_gpu_memory()
    if used is not None:
        print(f"\n📊 Updated GPU Memory: {used} MB / {total} MB ({percent:.1f}% used)")
    
    print("\n✅ Done! GPU memory should be freed.")
    print("=" * 60)

if __name__ == "__main__":
    main()

