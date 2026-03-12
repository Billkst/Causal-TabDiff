"""
测试新的 landmark wrappers
"""
import torch
import sys
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tsdiff_landmark_wrapper import TSDiffLandmarkWrapper
from baselines.stasy_landmark_wrapper import STaSyLandmarkWrapper


def test_wrapper(name, wrapper_class, train_loader, device):
    print(f"\n{'='*60}")
    print(f"测试 {name}")
    print(f"{'='*60}")
    
    sample = next(iter(train_loader))
    seq_len = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    
    try:
        wrapper = wrapper_class(seq_len, feature_dim)
        print(f"✓ Wrapper 创建")
        
        print(f"[1/3] 训练 (1 epoch, 2 batches)...")
        
        class LimitedLoader:
            def __init__(self, loader, max_batches=2):
                self.loader = loader
                self.max_batches = max_batches
            def __iter__(self):
                for i, batch in enumerate(self.loader):
                    if i >= self.max_batches:
                        break
                    yield batch
        
        limited_loader = LimitedLoader(train_loader, 2)
        wrapper.fit(limited_loader, epochs=1, device=device)
        print(f"✓ 训练完成")
        
        print(f"[2/3] 生成 (N=10)...")
        X_syn, Y_syn = wrapper.sample(10, device)
        print(f"✓ 生成完成: X={X_syn.shape}, Y={Y_syn.shape}")
        
        print(f"[3/3] 检查...")
        print(f"✓ Y unique: {torch.unique(Y_syn.cpu())[:5].tolist()}")
        
        return "✅ 成功"
        
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return "❌ 失败"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=42)
    train_loader, _, _ = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=32)
    
    results = {}
    results['TSDiff'] = test_wrapper('TSDiff', TSDiffLandmarkWrapper, train_loader, device)
    results['STaSy'] = test_wrapper('STaSy', STaSyLandmarkWrapper, train_loader, device)
    
    print(f"\n{'='*60}")
    print(f"结果")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{name}: {status}")


if __name__ == '__main__':
    main()
