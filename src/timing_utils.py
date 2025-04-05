import time
import functools
import logging
from collections import defaultdict
import os
import json
from datetime import datetime

class TimingStats:
    """
    A class to collect and manage timing statistics across the application.
    """
    def __init__(self, log_dir=None):
        self.stats = defaultdict(list)
        self.current_operation = None
        self.start_time = None
        self.log_dir = log_dir or "experiments"
        self.experiment_name = None
        self.logger = logging.getLogger("TimingStats")
        # For the new timer functionality
        self.timer_stats = {}
        
    def start_operation(self, operation_name):
        """Start timing an operation."""
        self.current_operation = operation_name
        self.start_time = time.time()
        
    def end_operation(self, operation_name=None):
        """End timing an operation and record the elapsed time."""
        if self.start_time is None:
            # Instead of warning, just create a minimal placeholder timing
            if operation_name is None:
                operation_name = self.current_operation or "unknown_operation"
            elapsed = 0.0  # Use zero as placeholder time
            self.stats[operation_name].append(elapsed)
            self.current_operation = None
            return elapsed
            
        if operation_name is None:
            operation_name = self.current_operation
            
        if operation_name is None:
            # If we still don't have an operation name, use a default
            operation_name = "unnamed_operation"
            
        elapsed = time.time() - self.start_time
        self.stats[operation_name].append(elapsed)
        self.current_operation = None
        self.start_time = None
        return elapsed
        
    def get_stats(self):
        """Get the current timing statistics."""
        result = {}
        for op, times in self.stats.items():
            if times:
                result[op] = {
                    "count": len(times),
                    "total": sum(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times)
                }
        
        # Also include timer stats from the Timer class
        for op, data in self.timer_stats.items():
            if data["count"] > 0:
                result[op] = {
                    "count": data["count"],
                    "total": data["total_time"],
                    "avg": data["total_time"] / data["count"],
                    "min": data["min_time"],
                    "max": data["max_time"]
                }
                
        return result
        
    def set_experiment_name(self, name):
        """Set the experiment name for logging."""
        self.experiment_name = name
        
        # Configure the logger for this experiment
        global timing_logger
        if timing_logger.handlers:
            for handler in timing_logger.handlers[:]:
                timing_logger.removeHandler(handler)
        
        timing_logger.setLevel(logging.DEBUG)
        # Use os.path.join for consistent path handling
        experiment_dir = os.path.join(self.log_dir, name)
        log_file = os.path.join(experiment_dir, "debug_log.txt")
        os.makedirs(experiment_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        timing_logger.addHandler(file_handler)
        
    def save_stats(self):
        """Save the timing statistics to a JSON file."""
        if not self.experiment_name:
            self.logger.warning("No experiment name set, using timestamp")
            self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        stats = self.get_stats()
        # Fix the path to avoid duplicate experiment name
        experiment_dir = os.path.join(self.log_dir, self.experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        file_path = os.path.join(experiment_dir, "timing_stats.json")
        
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        self.logger.info(f"Timing statistics saved to {file_path}")
        return file_path
    
    # Methods to support the Timer class and timed_operation decorator
    def register_timer_stats(self, label):
        """Register a new timer with the given label"""
        if label not in self.timer_stats:
            self.timer_stats[label] = {
                "total_time": 0.0, 
                "count": 0, 
                "min_time": float('inf'), 
                "max_time": 0.0
            }
    
    def update_timer_stats(self, label, elapsed):
        """Update stats for a timer with new timing data"""
        if label not in self.timer_stats:
            self.register_timer_stats(label)
            
        stat_entry = self.timer_stats[label]
        stat_entry["total_time"] += elapsed
        stat_entry["count"] += 1
        stat_entry["min_time"] = min(stat_entry["min_time"], elapsed)
        stat_entry["max_time"] = max(stat_entry["max_time"], elapsed)
        
    def dump_timing_stats(self, output_file=None):
        """Dump all timing statistics to a file or return as string."""
        lines = ["===== TIMING STATISTICS ====="]
        
        # Get combined stats
        all_stats = self.get_stats()
        
        # Sort operations by total time
        sorted_stats = sorted(all_stats.items(), key=lambda x: x[1]["total"], reverse=True)
        
        for name, stats in sorted_stats:
            lines.append(f"{name}: {stats['avg']:.6f}s avg, {stats['total']:.2f}s total, "
                        f"{stats['count']} calls, min={stats['min']:.6f}s, max={stats['max']:.6f}s")
        
        output = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'a') as f:
                f.write(output + "\n\n")
        
        timing_logger.info("\n" + output)
        return output

# Global timing stats instance
timing_stats = TimingStats()

# Configure logging for timing data
timing_logger = logging.getLogger('timing_utils')

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, label, debug=False):
        self.label = label
        self.debug = debug
        self.elapsed = 0.0
        
        # Initialize stats entry
        timing_stats.register_timer_stats(label)
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def get_elapsed(self):
        """Get the current elapsed time without exiting the context."""
        return time.time() - self.start
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed = self.end - self.start
        
        # Update global timing statistics
        timing_stats.update_timer_stats(self.label, self.elapsed)
        
        if self.debug:
            print(f"[Timer] {self.label}: {self.elapsed:.6f}s")
            
        # Log to debug file at a reasonable frequency
        stat_entry = timing_stats.timer_stats[self.label]
        if stat_entry["count"] % 100 == 0:
            avg_time = stat_entry["total_time"] / stat_entry["count"]
            timing_logger.debug(f"{self.label}: {self.elapsed:.6f}s (avg: {avg_time:.6f}s, calls: {stat_entry['count']})")

def timed_operation(operation_name):
    """Decorator to time function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get debug flag from kwargs if available
            debug = kwargs.get('debug', False)
            
            # Initialize stats entry
            timing_stats.register_timer_stats(operation_name)
            
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Update global timing statistics
            timing_stats.update_timer_stats(operation_name, elapsed)
            
            if debug:
                print(f"[Timer] {operation_name}: {elapsed:.6f}s")
                
            # Log to debug file at a reasonable frequency
            stat_entry = timing_stats.timer_stats[operation_name]
            if stat_entry["count"] % 100 == 0:
                avg_time = stat_entry["total_time"] / stat_entry["count"]
                timing_logger.debug(f"{operation_name}: {elapsed:.6f}s (avg: {avg_time:.6f}s, calls: {stat_entry['count']})")
                
            return result
        return wrapper
    return decorator

def print_timing_summary():
    """Print a summary of all timing statistics."""
    stats = timing_stats.get_stats()
    
    print("\n===== TIMING SUMMARY =====")
    for op, data in stats.items():
        print(f"{op}:")
        print(f"  Count: {data['count']}")
        print(f"  Total: {data['total']:.6f}s")
        print(f"  Avg:   {data['avg']:.6f}s")
        print(f"  Min:   {data['min']:.6f}s")
        print(f"  Max:   {data['max']:.6f}s")
    print("=========================\n") 

# For backward compatibility
dump_timing_stats = timing_stats.dump_timing_stats 