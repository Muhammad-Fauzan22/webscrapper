#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYSTEM PERFORMANCE MONITOR
Version: 2.1.0
Created: 2025-07-15
Author: AI Infrastructure Team
"""

import psutil
import platform
import time
import logging
from datetime import datetime

logger = logging.getLogger("SystemMonitor")

class SystemMonitor:
    def __init__(self, history_size=10):
        self.history = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network': [],
            'timestamps': []
        }
        self.history_size = history_size
        self.start_time = datetime.now()
        self.last_metrics = {}
        self.anomalies = []
        
    def cpu_usage(self, interval=1.0) -> float:
        """Get current CPU usage with smoothing"""
        try:
            # Take two measurements to calculate usage
            usage1 = psutil.cpu_percent(interval=None)
            time.sleep(interval)
            usage2 = psutil.cpu_percent(interval=None)
            avg_usage = (usage1 + usage2) / 2.0
            
            # Update history
            self._update_history('cpu', avg_usage)
            return avg_usage
        except Exception as e:
            logger.error(f"CPU monitoring failed: {str(e)}")
            return 0.0

    def memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            usage = psutil.virtual_memory().percent
            self._update_history('memory', usage)
            return usage
        except Exception as e:
            logger.error(f"Memory monitoring failed: {str(e)}")
            return 0.0

    def disk_usage(self, path='/') -> float:
        """Get current disk usage percentage"""
        try:
            usage = psutil.disk_usage(path).percent
            self._update_history('disk', usage)
            return usage
        except Exception as e:
            logger.error(f"Disk monitoring failed: {str(e)}")
            return 0.0

    def network_usage(self) -> dict:
        """Get network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            usage = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            self._update_history('network', usage)
            return usage
        except Exception as e:
            logger.error(f"Network monitoring failed: {str(e)}")
            return {}

    def get_all_metrics(self) -> dict:
        """Get comprehensive system metrics"""
        return {
            'cpu': self.cpu_usage(),
            'memory': self.memory_usage(),
            'disk': self.disk_usage(),
            'network': self.network_usage(),
            'uptime': self.get_uptime(),
            'os': self.get_os_info(),
            'processes': self.get_process_count(),
            'temperatures': self.get_temperatures(),
            'anomalies': self.anomalies
        }

    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return (datetime.now() - self.start_time).total_seconds()

    def get_os_info(self) -> dict:
        """Get operating system information"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }

    def get_process_count(self) -> int:
        """Get number of running processes"""
        try:
            return len(psutil.pids())
        except:
            return 0

    def get_temperatures(self) -> dict:
        """Get system temperatures if available"""
        try:
            temps = psutil.sensors_temperatures()
            return {k: [t.current for t in v] for k, v in temps.items()}
        except:
            return {}

    def detect_anomalies(self) -> list:
        """Detect performance anomalies based on historical data"""
        anomalies = []
        
        # CPU anomaly detection
        if len(self.history['cpu']) > 5:
            avg_cpu = sum(self.history['cpu'][-5:]) / 5
            if self.history['cpu'][-1] > avg_cpu * 1.5:
                anomalies.append({
                    'type': 'CPU_SPIKE',
                    'value': self.history['cpu'][-1],
                    'avg': avg_cpu,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Memory leak detection
        if len(self.history['memory']) > 10:
            trend = self._calculate_trend(self.history['memory'][-10:])
            if trend > 0.5:  # Increasing by 0.5% per measurement
                anomalies.append({
                    'type': 'MEMORY_LEAK',
                    'trend': f"{trend:.2f}%/measurement",
                    'current': self.history['memory'][-1],
                    'timestamp': datetime.now().isoformat()
                })
        
        self.anomalies.extend(anomalies)
        return anomalies

    def _update_history(self, metric: str, value):
        """Update metric history with rolling window"""
        self.history[metric].append(value)
        self.history['timestamps'].append(time.time())
        
        # Maintain history size
        if len(self.history[metric]) > self.history_size:
            self.history[metric].pop(0)
            
        if len(self.history['timestamps']) > self.history_size:
            self.history['timestamps'].pop(0)

    def _calculate_trend(self, values: list) -> float:
        """Calculate linear trend of values"""
        if len(values) < 2:
            return 0.0
            
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x_sq_sum = sum(i*i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum**2)
        return slope
