"""
效率指标收集模块
"""
import time
import datetime
import importlib
import importlib.util
try:
    _psutil_spec = importlib.util.find_spec('psutil')
    if _psutil_spec is not None:
        psutil = importlib.import_module('psutil')
    else:
        psutil = None
except ImportError:
    psutil = None
HAS_PSUTIL = psutil is not None
import os
import torch
import numpy as np
from contextlib import contextmanager


class EfficiencyTracker:
    """训练和推理效率追踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        if psutil is not None:
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
        self.metrics['train_wall_clock_sec'] = elapsed
        
        if torch.cuda.is_available():
            self.metrics['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.metrics['gpu_memory_mb_peak'] = self.metrics['peak_gpu_memory_mb']
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
            self.metrics['num_parameters_total'] = total
            self.metrics['num_parameters_trainable'] = trainable
        else:
            self.metrics['total_params'] = 'N/A'
            self.metrics['trainable_params'] = 'N/A'
            self.metrics['num_parameters_total'] = 'N/A'
            self.metrics['num_parameters_trainable'] = 'N/A'

    def set_metadata(self, model_name, layer, seed, experiment_type, command=""):
        self.metrics['model_name'] = model_name
        self.metrics['layer'] = layer
        self.metrics['seed'] = seed
        self.metrics['experiment_type'] = experiment_type
        self.metrics['command'] = command
        self.metrics['timestamp'] = datetime.datetime.now().isoformat()

    def set_gpu_info(self):
        if torch.cuda.is_available():
            self.metrics['gpu_name'] = torch.cuda.get_device_name(0)
        else:
            self.metrics['gpu_name'] = 'CPU'

    def set_output_paths(self, output_file, log_file):
        self.metrics['output_file'] = output_file
        self.metrics['log_file'] = log_file

    def set_epochs(self, num_epochs_actual, average_epoch_time_sec):
        self.metrics['num_epochs_actual'] = num_epochs_actual
        self.metrics['average_epoch_time_sec'] = average_epoch_time_sec

    @contextmanager
    def track_phase_time(self, phase_name):
        start = time.time()
        yield
        elapsed = time.time() - start
        self.metrics[f'{phase_name}_wall_clock_sec'] = elapsed

    def measure_inference_latency(self, forward_fn, batch_size=64, n_iterations=1000, warmup_steps=20):
        latencies = []
        use_cuda_sync = torch.cuda.is_available()

        with torch.no_grad():
            for _ in range(warmup_steps):
                forward_fn()
                if use_cuda_sync:
                    torch.cuda.synchronize()

            for _ in range(n_iterations):
                if use_cuda_sync:
                    torch.cuda.synchronize()
                t0 = time.time()
                forward_fn()
                if use_cuda_sync:
                    torch.cuda.synchronize()
                latencies.append(time.time() - t0)

        latency_arr = np.array(latencies, dtype=np.float64)
        mean_latency = float(np.mean(latency_arr))

        self.metrics['inference_latency_ms_per_batch'] = mean_latency * 1000
        self.metrics['inference_latency_ms_per_sample'] = (mean_latency / batch_size) * 1000
        self.metrics['inference_batch_size'] = batch_size
        self.metrics['inference_n_samples'] = n_iterations * batch_size
        self.metrics['inference_warmup_steps'] = warmup_steps

        self.metrics['inference_latency_std_ms'] = float(np.std(latency_arr)) * 1000
        self.metrics['inference_latency_p50_ms'] = float(np.percentile(latency_arr, 50)) * 1000
        self.metrics['inference_latency_p95_ms'] = float(np.percentile(latency_arr, 95)) * 1000
        self.metrics['inference_latency_p99_ms'] = float(np.percentile(latency_arr, 99)) * 1000
        self.metrics['inference_throughput_samples_per_sec'] = batch_size / mean_latency if mean_latency > 0 else float('inf')

    def validate_schema(self):
        required_fields = [
            'model_name', 'layer', 'seed', 'experiment_type',
            'num_parameters_total', 'num_parameters_trainable',
            'train_wall_clock_sec', 'total_wall_clock_sec',
            'average_epoch_time_sec', 'num_epochs_actual',
            'inference_latency_ms_per_sample', 'inference_latency_ms_per_batch',
            'inference_batch_size', 'inference_n_samples', 'inference_warmup_steps',
            'gpu_memory_mb_peak', 'gpu_name',
            'output_file', 'log_file', 'command', 'timestamp'
        ]

        alias_mapping = {
            'train_wall_clock_sec': 'total_training_wall_clock_sec',
            'total_training_wall_clock_sec': 'train_wall_clock_sec',
            'num_parameters_total': 'total_params',
            'total_params': 'num_parameters_total',
            'num_parameters_trainable': 'trainable_params',
            'trainable_params': 'num_parameters_trainable',
            'gpu_memory_mb_peak': 'peak_gpu_memory_mb',
            'peak_gpu_memory_mb': 'gpu_memory_mb_peak',
        }

        missing = []
        for field in required_fields:
            alias = alias_mapping.get(field)
            if field not in self.metrics and (alias is None or alias not in self.metrics):
                missing.append(field)

        is_valid = len(missing) == 0
        if is_valid:
            print("Schema validation: ✅ PASS")
        else:
            print(f"Schema validation: ❌ MISSING: {missing}")
        return is_valid, missing
    
    def set_epoch_time(self, avg_epoch_time):
        """记录平均 epoch 时间"""
        self.metrics['average_epoch_time_sec'] = avg_epoch_time
    
    def get_metrics(self):
        """返回所有效率指标"""
        return self.metrics.copy()
    
    def save_json(self, filepath):
        """保存效率指标到 JSON 文件"""
        import json

        if 'total_training_wall_clock_sec' in self.metrics and 'train_wall_clock_sec' not in self.metrics:
            self.metrics['train_wall_clock_sec'] = self.metrics['total_training_wall_clock_sec']
        if 'total_params' in self.metrics and 'num_parameters_total' not in self.metrics:
            self.metrics['num_parameters_total'] = self.metrics['total_params']
        if 'trainable_params' in self.metrics and 'num_parameters_trainable' not in self.metrics:
            self.metrics['num_parameters_trainable'] = self.metrics['trainable_params']
        if 'peak_gpu_memory_mb' in self.metrics and 'gpu_memory_mb_peak' not in self.metrics:
            self.metrics['gpu_memory_mb_peak'] = self.metrics['peak_gpu_memory_mb']

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
