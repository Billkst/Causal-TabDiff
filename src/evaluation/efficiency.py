"""
效率指标收集模块
"""
import time
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import os
import torch
import numpy as np
from contextlib import contextmanager


class EfficiencyTracker:
    """训练和推理效率追踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        if HAS_PSUTIL:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
        
    @contextmanager
    def track_training(self):
        """追踪训练阶段"""
        self.start_time = time.time()
        if HAS_PSUTIL and self.process:
            start_mem = self.process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        yield
        
        elapsed = time.time() - self.start_time
        if HAS_PSUTIL and self.process:
            end_mem = self.process.memory_info().rss / 1024 / 1024
            self.metrics['peak_cpu_ram_mb'] = end_mem
        
        self.metrics['total_training_wall_clock_sec'] = elapsed
        
        if torch.cuda.is_available():
            self.metrics['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.metrics['total_training_gpu_hours'] = elapsed / 3600
            self.metrics['device_type'] = 'cuda'
        else:
            self.metrics['total_training_cpu_hours'] = elapsed / 3600
            self.metrics['device_type'] = 'cpu'
    
    @contextmanager
    def track_inference(self, n_samples):
        """追踪推理阶段"""
        start = time.time()
        yield
        elapsed = time.time() - start
        
        self.metrics['inference_latency_ms_per_sample'] = (elapsed / n_samples) * 1000
        self.metrics['throughput_samples_per_sec'] = n_samples / elapsed
    
    def set_model_size(self, model):
        """记录模型规模"""
        if hasattr(model, 'parameters'):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.metrics['total_params'] = total
            self.metrics['trainable_params'] = trainable
        else:
            self.metrics['total_params'] = 'N/A'
            self.metrics['trainable_params'] = 'N/A'
    
    def set_epoch_time(self, avg_epoch_time):
        """记录平均 epoch 时间"""
        self.metrics['average_epoch_time_sec'] = avg_epoch_time
    
    def get_metrics(self):
        """返回所有效率指标"""
        return self.metrics.copy()
    
    def save_json(self, filepath):
        """保存效率指标到 JSON 文件"""
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
