import os
import time
import psutil
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
from threading import Thread, Event

logger = logging.getLogger(__name__)

class SystemResourceTracker:
    """
    Tracks CPU and memory usage during pipeline execution.
    
    This class provides methods to monitor system resources (CPU and RAM) 
    during the execution of a Haystack pipeline. It can be used to identify
    performance bottlenecks and resource-intensive components.
    """
    
    def __init__(
        self,
        sampling_interval: float = 0.5,
        track_process: bool = True,
        include_children: bool = True
    ):
        """
        Initialize the system resource tracker.
        
        Args:
            sampling_interval: Time in seconds between resource usage samples.
            track_process: Whether to track the current process specifically 
                         (in addition to system-wide metrics).
            include_children: Whether to include child processes in process metrics.
        """
        self.sampling_interval = sampling_interval
        self.track_process = track_process
        self.include_children = include_children
        self.current_process = psutil.Process(os.getpid()) if track_process else None
        
        # Storage for metrics
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        
        # Thread control
        self._stop_event = Event()
        self._tracking_thread = None
        
        logger.info(f"SystemResourceTracker initialized with sampling interval: {sampling_interval}s")
    
    def start_tracking(self):
        """Start tracking system resources in a background thread."""
        if self._tracking_thread and self._tracking_thread.is_alive():
            logger.warning("Resource tracking is already running")
            return
        
        self._stop_event.clear()
        self._tracking_thread = Thread(target=self._track_resources, daemon=True)
        self._tracking_thread.start()
        logger.info("Started resource tracking")
    
    def stop_tracking(self):
        """Stop tracking system resources."""
        if not self._tracking_thread or not self._tracking_thread.is_alive():
            logger.warning("Resource tracking is not running")
            return
        
        self._stop_event.set()
        self._tracking_thread.join(timeout=2*self.sampling_interval)
        if self._tracking_thread.is_alive():
            logger.warning("Resource tracking thread did not terminate correctly")
        else:
            logger.info("Stopped resource tracking")
    
    def _track_resources(self):
        """Background thread function to collect resource metrics at regular intervals."""
        while not self._stop_event.is_set():
            try:
                # Get current timestamp
                now = datetime.now()
                
                # Get system-wide CPU usage
                system_cpu_percent = psutil.cpu_percent(interval=None)
                
                # Get system-wide memory usage
                system_memory = psutil.virtual_memory()
                
                # Get process-specific metrics if requested
                process_cpu_percent = None
                process_memory_mb = None
                
                if self.track_process and self.current_process:
                    # Process CPU percentage
                    process_cpu_percent = self.current_process.cpu_percent(interval=None)
                    
                    # Process memory usage in MB
                    process_memory = self.current_process.memory_info()
                    process_memory_mb = process_memory.rss / (1024 * 1024)  # Convert bytes to MB
                
                # Store the measurements
                self.timestamps.append(now)
                self.cpu_samples.append({
                    'system': system_cpu_percent,
                    'process': process_cpu_percent
                })
                self.memory_samples.append({
                    'system_percent': system_memory.percent,
                    'system_used_gb': system_memory.used / (1024**3),  # Convert bytes to GB
                    'system_available_gb': system_memory.available / (1024**3),  # Convert bytes to GB
                    'process_mb': process_memory_mb
                })
                
            except Exception as e:
                logger.error(f"Error collecting resource metrics: {e}")
            
            # Sleep until next sample
            time.sleep(self.sampling_interval)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of collected resource metrics.
        
        Returns:
            Dict with summary statistics of CPU and memory usage
        """
        if not self.cpu_samples or not self.memory_samples:
            return {"error": "No metrics collected"}
        
        # Convert to pandas DataFrame for easier analysis
        cpu_df = pd.DataFrame(self.cpu_samples)
        memory_df = pd.DataFrame(self.memory_samples)
        
        # Calculate summary statistics
        summary = {
            "cpu": {
                "system": {
                    "mean": cpu_df['system'].mean(),
                    "max": cpu_df['system'].max(),
                    "min": cpu_df['system'].min()
                }
            },
            "memory": {
                "system_percent": {
                    "mean": memory_df['system_percent'].mean(),
                    "max": memory_df['system_percent'].max(),
                    "min": memory_df['system_percent'].min()
                },
                "system_used_gb": {
                    "mean": memory_df['system_used_gb'].mean(),
                    "max": memory_df['system_used_gb'].max(),
                    "min": memory_df['system_used_gb'].min()
                }
            },
            "duration_seconds": (self.timestamps[-1] - self.timestamps[0]).total_seconds(),
            "samples_count": len(self.timestamps)
        }
        
        # Add process-specific metrics if available
        if self.track_process and 'process' in cpu_df and not cpu_df['process'].isna().all():
            summary["cpu"]["process"] = {
                "mean": cpu_df['process'].mean(),
                "max": cpu_df['process'].max(),
                "min": cpu_df['process'].min()
            }
            
        if self.track_process and 'process_mb' in memory_df and not memory_df['process_mb'].isna().all():
            summary["memory"]["process_mb"] = {
                "mean": memory_df['process_mb'].mean(),
                "max": memory_df['process_mb'].max(),
                "min": memory_df['process_mb'].min()
            }
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert collected metrics to a pandas DataFrame.
        
        Returns:
            DataFrame containing all collected metrics with timestamps
        """
        if not self.cpu_samples or not self.memory_samples:
            return pd.DataFrame()
        
        # Combine all metrics into a single list of dictionaries
        data = []
        for i, timestamp in enumerate(self.timestamps):
            entry = {"timestamp": timestamp}
            entry.update({f"cpu_{k}": v for k, v in self.cpu_samples[i].items()})
            entry.update({f"memory_{k}": v for k, v in self.memory_samples[i].items()})
            data.append(entry)
        
        return pd.DataFrame(data)
    
    def to_csv(self, file_path: str) -> None:
        """
        Save collected metrics to a CSV file.
        
        Args:
            file_path: Path to save the CSV file
        """
        df = self.to_dataframe()
        if df.empty:
            logger.warning("No metrics to save")
            return
        
        df.to_csv(file_path, index=False)
        logger.info(f"Saved resource metrics to {file_path}")
    
    def print_summary(self) -> None:
        """Print a summary of the collected resource metrics to the console."""
        summary = self.get_metrics_summary()
        
        if "error" in summary:
            print(f"Error: {summary['error']}")
            return
        
        print("\n===== System Resource Usage Summary =====")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Samples: {summary['samples_count']}")
        
        print("\n--- CPU Usage ---")
        print(f"System CPU: {summary['cpu']['system']['mean']:.2f}% (avg), "
              f"{summary['cpu']['system']['max']:.2f}% (max)")
        
        if "process" in summary["cpu"]:
            print(f"Process CPU: {summary['cpu']['process']['mean']:.2f}% absolut, {summary['cpu']['process']['mean']/psutil.cpu_count()*100:.2f}% relativ")
        
        print("\n--- Memory Usage ---")
        print(f"System Memory: {summary['memory']['system_percent']['mean']:.2f}% (avg), "
              f"{summary['memory']['system_percent']['max']:.2f}% (max)")
        print(f"System Memory Used: {summary['memory']['system_used_gb']['mean']:.2f} GB (avg), "
              f"{summary['memory']['system_used_gb']['max']:.2f} GB (max)")
        
        if "process_mb" in summary["memory"]:
            print(f"Process Memory: {summary['memory']['process_mb']['mean']:.2f} MB (avg), "
                  f"{summary['memory']['process_mb']['max']:.2f} MB (max)")
        
        print("===========================================\n")


def track_pipeline_resources(func):
    """
    Decorator to track system resources during pipeline execution.
    
    This decorator can be used to wrap any function that executes a Haystack pipeline
    to collect resource usage metrics during its execution.
    
    Example:
        @track_pipeline_resources
        def run_my_pipeline(pipeline, query):
            return pipeline.run(query=query)
    """
    def wrapper(*args, **kwargs):
        # Create and start the tracker
        tracker = SystemResourceTracker()
        tracker.start_tracking()
        
        try:
            # Run the function
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop tracking and print summary
            tracker.stop_tracking()
            tracker.print_summary()
            
            # Save metrics to CSV if requested
            csv_path = kwargs.get('resource_csv_path')
            if csv_path:
                try:
                    tracker.to_csv(csv_path)
                except:
                    pass
    
    return wrapper


class PipelineResourceMonitor:
    """
    Monitor system resources during pipeline execution.
    
    This class allows for monitoring Haystack pipelines by wrapping the pipeline.run method
    to track system resource usage during execution.
    
    Example:
        # Create a monitor for your pipeline
        monitor = PipelineResourceMonitor(pipeline)
        
        # Run the pipeline with monitoring
        result = monitor.run(data={"query": "What is the capital of France?"})
        
        # Get the resource usage summary
        monitor.print_summary()
    """
    
    def __init__(self, pipeline, sampling_interval: float = 0.5):
        """
        Initialize the pipeline resource monitor.
        
        Args:
            pipeline: Haystack pipeline to monitor
            sampling_interval: Time in seconds between resource usage samples
        """
        self.pipeline = pipeline
        self.tracker = SystemResourceTracker(sampling_interval=sampling_interval)
        self.last_run_metrics = None
    
    def run(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the pipeline with resource monitoring.
        
        Args:
            data: Input data for the pipeline
            **kwargs: Additional arguments to pass to pipeline.run
            
        Returns:
            Result of pipeline execution
        """
        # Start tracking resources
        self.tracker = SystemResourceTracker()  # Create a new tracker for each run
        self.tracker.start_tracking()
        
        try:
            # Run the pipeline
            result = self.pipeline.run(data, **kwargs)
            return result
        finally:
            # Stop tracking and save metrics
            self.tracker.stop_tracking()
            self.last_run_metrics = self.tracker.get_metrics_summary()
    
    def print_summary(self) -> None:
        """Print a summary of the resource usage during the last pipeline run."""
        if self.tracker:
            self.tracker.print_summary()
        else:
            print("No monitoring data available. Run the pipeline first.")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of resource metrics from the last pipeline run.
        
        Returns:
            Dict with summary statistics of CPU and memory usage
        """
        if self.tracker:
            return self.tracker.get_metrics_summary()
        return {"error": "No monitoring data available"}
    
    def save_metrics(self, csv_file: str = None) -> None:
        """
        Save the metrics from the last pipeline run.
        
        Args:
            csv_file: Optional file path to save metrics CSV
        """
        if not self.tracker:
            print("No monitoring data available. Run the pipeline first.")
            return
            
        if csv_file:
            self.tracker.to_csv(csv_file) 