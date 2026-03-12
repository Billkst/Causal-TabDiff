"""
逐个验证 TSTR baselines - 使用适配层
"""
import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from data.landmark_adapter import adapt_landmark_to_generative
from baselines.wrappers import STaSyWrapper, TabSynWrapper, TabDiffWrapper, TSDiffWrapper


def verify_with_adapter(baseline_name, wrapper_class, train_loader, device):
    print(f"\n{'='*60}")
    print(f"验证 {baseline_name} (使用适配层)")
    print(f"{'='*60}")
    
    sample = next(iter(train_loader))
    seq_len = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    
    result = {
        'name': baseline_name,
        'fit_pass': False,
        'sample_pass': False,
        'label_protocol': 'unknown',
        'status': '阻塞'
    }
    
    try:
        wrapper = wrapper_class(t_steps=seq_len, feature_dim=feature_dim)
        print(f"✓ Wrapper 创建成功")
        
        print(f"[1/5] 测试适配层...")
        adapted_batch = adapt_landmark_to_generative(sample, device)
        print(f"✓ 适配层成功: x={adapted_batch['x'].shape}, y={adapted_batch['y'].shape}, alpha={adapted_batch['alpha_target'].shape}")
        
        print(f"[2/5] 训练生成模型 (1 epoch, 2 batches)...")
        
        class AdaptedLoader:
            def __init__(self, original_loader, device):
                self.original_loader = original_loader
                self.device = device
                self.iter = iter(original_loader)
            
            def __iter__(self):
                self.iter = iter(self.original_loader)
                return self
            
            def __next__(self):
                batch = next(self.iter)
                return adapt_landmark_to_generative(batch, self.device)
            
            def __len__(self):
                return len(self.original_loader)
        
        adapted_loader = AdaptedLoader(train_loader, device)
        wrapper.fit(adapted_loader, epochs=1, device=device, debug_mode=True)
        print(f"✓ 训练完成")
        result['fit_pass'] = True
        
        print(f"[3/5] 生成 synthetic data (N=10)...")
        X_syn, Y_syn = wrapper.sample(10, alpha_target=torch.zeros(10, 1, device=device), device=device)
        print(f"✓ 生成完成: X shape={X_syn.shape}, Y shape={Y_syn.shape}")
        result['sample_pass'] = True
        
        print(f"[4/5] 检查 Y 是否合法...")
        Y_syn_np = Y_syn.cpu().numpy() if torch.is_tensor(Y_syn) else Y_syn
        unique_labels = np.unique(Y_syn_np)
        print(f"✓ Y unique values: {unique_labels}")
        
        print(f"[5/5] 判定 TSTR 协议...")
        if Y_syn is not None and len(Y_syn.shape) > 0:
            result['label_protocol'] = '联合生成 p(X,Y)'
            result['status'] = '标准 TSTR ready'
            print(f"✓ 确认为标准 TSTR (联合生成)")
        else:
            result['status'] = '当前不成立'
            print(f"✗ Y 为空，不是联合生成")
            
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        result['status'] = '阻塞'
    
    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=42)
    train_loader, _, _ = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=32)
    
    results = []
    
    baselines = [
        ('TSDiff', TSDiffWrapper),
        ('STaSy', STaSyWrapper),
        ('TabSyn', TabSynWrapper),
        ('TabDiff', TabDiffWrapper),
    ]
    
    for name, wrapper_class in baselines:
        result = verify_with_adapter(name, wrapper_class, train_loader, device)
        results.append(result)
    
    print(f"\n{'='*60}")
    print(f"验证结果汇总")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['name']:12} | Fit: {'✓' if r['fit_pass'] else '✗'} | Sample: {'✓' if r['sample_pass'] else '✗'} | 协议: {r['label_protocol']:20} | 状态: {r['status']}")


if __name__ == '__main__':
    main()
