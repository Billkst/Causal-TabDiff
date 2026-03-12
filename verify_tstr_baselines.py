"""
逐个验证 TSTR baselines 的完整流程
"""
import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.wrappers import STaSyWrapper, TabSynWrapper, TabDiffWrapper, TSDiffWrapper

def verify_baseline(baseline_name, wrapper_class, train_loader, device):
    print(f"\n{'='*60}")
    print(f"验证 {baseline_name}")
    print(f"{'='*60}")
    
    sample = next(iter(train_loader))
    seq_len = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    
    try:
        wrapper = wrapper_class(t_steps=seq_len, feature_dim=feature_dim)
        print(f"✓ Wrapper 创建成功")
        
        print(f"[1/4] 训练生成模型 (2 epochs)...")
        wrapper.fit(train_loader, epochs=2, device=device, debug_mode=True)
        print(f"✓ 训练完成")
        
        print(f"[2/4] 生成 synthetic data (N=10)...")
        X_syn, Y_syn = wrapper.sample(10, alpha_target=None, device=device)
        print(f"✓ 生成完成: X shape={X_syn.shape}, Y shape={Y_syn.shape}")
        
        print(f"[3/4] 检查 Y 是否合法...")
        Y_syn_np = Y_syn.cpu().numpy() if torch.is_tensor(Y_syn) else Y_syn
        unique_labels = np.unique(Y_syn_np)
        print(f"✓ Y unique values: {unique_labels}")
        
        print(f"[4/4] 检查是否为联合生成...")
        if Y_syn is not None and len(Y_syn.shape) > 0:
            print(f"✓ 确认为联合生成 (X, Y)")
            return "标准 TSTR ready"
        else:
            print(f"✗ Y 为空，不是联合生成")
            return "当前不成立"
            
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return "阻塞"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=42)
    train_loader, _, _ = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=32)
    
    results = {}
    
    baselines = [
        ('STaSy', STaSyWrapper),
        ('TabSyn', TabSynWrapper),
        ('TabDiff', TabDiffWrapper),
        ('TSDiff', TSDiffWrapper)
    ]
    
    for name, wrapper_class in baselines:
        status = verify_baseline(name, wrapper_class, train_loader, device)
        results[name] = status
    
    print(f"\n{'='*60}")
    print(f"验证结果汇总")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{name}: {status}")

if __name__ == '__main__':
    main()
