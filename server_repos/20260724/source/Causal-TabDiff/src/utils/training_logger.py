"""最小化训练日志工具 - 满足实时输出和文件记录要求"""
import os
import sys
from pathlib import Path


class TrainingLogger:
    """统一的训练日志记录器，支持终端和文件同时输出"""
    
    def __init__(self, log_dir='logs', script_name='training'):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            script_name: 脚本名称（用于日志文件名）
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        self.log_file = self.log_dir / f"{script_name}.log"
        
        # 打开日志文件，使用行缓冲模式
        self.log_fp = open(self.log_file, 'w', buffering=1, encoding='utf-8')
    
    def log(self, message):
        """输出日志到终端和文件（带 flush）"""
        print(message, flush=True)
        self.log_fp.write(message + '\n')
        self.log_fp.flush()
    
    def epoch_log(self, epoch, total_epochs, seed, lr, train_loss, val_loss=None, 
                  best_val_metric=None, epoch_time=None, **kwargs):
        """
        输出 epoch 级别日志
        
        Args:
            epoch: 当前 epoch 编号
            total_epochs: 总 epoch 数
            seed: 随机种子
            lr: 学习率
            train_loss: 训练 loss
            val_loss: 验证 loss（可选）
            best_val_metric: 最佳验证指标（可选）
            epoch_time: epoch 耗时（可选）
            **kwargs: 其他指标（如 AUPRC、AUROC、F1 等）
        """
        parts = [
            f"Epoch {epoch}/{total_epochs}",
            f"Seed {seed}",
            f"LR {lr:.1e}",
            f"TrainLoss {train_loss:.4f}"
        ]
        
        if val_loss is not None:
            parts.append(f"ValLoss {val_loss:.4f}")
        
        if best_val_metric is not None:
            parts.append(f"BestValMetric {best_val_metric:.4f}")
        
        # 添加其他指标
        for key, val in kwargs.items():
            if val is not None:
                parts.append(f"{key} {val:.4f}")
        
        if epoch_time is not None:
            parts.append(f"Time {epoch_time:.1f}s")
        
        message = " | ".join(parts)
        self.log(message)
    
    def close(self):
        """关闭日志文件"""
        if self.log_fp:
            self.log_fp.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
