"""
统一 Baseline 测试脚本
覆盖所有当前可用的模型，输出结构化测试结果
"""
import torch
import sys
import traceback
sys.path.insert(0, 'src')

from data.data_module_landmark import load_and_split_data, create_dataloaders
from baselines.tsdiff_landmark_v2 import TSDiffLandmarkWrapper
from baselines.tabsyn_landmark_v2 import TabSynLandmarkWrapper
from baselines.tabdiff_landmark_v2 import TabDiffLandmarkWrapper
from baselines.stasy_landmark_v2 import STaSyLandmarkWrapper
from baselines.tslib_wrappers import iTransformerWrapper, TimeXerWrapper


def test_landmark_wrapper(name, wrapper_class, train_loader, device, epochs=2):
    """测试 landmark wrapper (TSDiff, STaSy, TabSyn, TabDiff)"""
    print(f"\n{'='*70}")
    print(f"测试 {name}")
    print(f"{'='*70}")
    
    sample = next(iter(train_loader))
    seq_len = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    
    result = {
        'model': name,
        'type': 'landmark_wrapper',
        'status': 'unknown',
        'error': None,
        'details': {}
    }
    
    try:
        # 1. 创建 wrapper
        wrapper = wrapper_class(seq_len, feature_dim)
        print(f"✓ [1/3] Wrapper 创建成功")
        result['details']['wrapper_created'] = True
        
        # 2. 训练
        print(f"  [2/3] 训练 ({epochs} epochs)...", flush=True)
        wrapper.fit(train_loader, epochs=epochs, device=device)
        print(f"✓ [2/3] 训练完成")
        result['details']['training_completed'] = True
        
        # 3. 生成样本
        print(f"  [3/3] 生成样本 (N=10)...", flush=True)
        X_syn, Y_syn = wrapper.sample(10, device)
        print(f"✓ [3/3] 生成完成: X={X_syn.shape}, Y={Y_syn.shape}")
        result['details']['sampling_completed'] = True
        result['details']['X_shape'] = str(X_syn.shape)
        result['details']['Y_shape'] = str(Y_syn.shape)
        result['details']['Y_unique'] = torch.unique(Y_syn.cpu())[:5].tolist()
        
        result['status'] = 'success'
        print(f"✅ {name} 测试通过")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"❌ {name} 测试失败: {str(e)}")
        traceback.print_exc()
    
    return result


def test_tslib_wrapper(name, model, train_loader, device, epochs=2):
    """测试 TSLib wrapper (iTransformer, TimeXer)"""
    print(f"\n{'='*70}")
    print(f"测试 {name}")
    print(f"{'='*70}")
    
    result = {
        'model': name,
        'type': 'tslib_wrapper',
        'status': 'unknown',
        'error': None,
        'details': {}
    }
    
    try:
        # 1. 模型创建
        model = model.to(device)
        print(f"✓ [1/3] 模型创建成功")
        result['details']['model_created'] = True
        
        # 2. 训练
        print(f"  [2/3] 训练 ({epochs} epochs)...", flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                x = batch['x'].to(device)
                traj_target = batch['trajectory_target'].to(device)
                traj_mask = batch['trajectory_valid_mask'].to(device)
                
                # Padding if needed
                if x.shape[1] < 3:
                    pad_len = 3 - x.shape[1]
                    x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
                
                optimizer.zero_grad()
                output = model(x)
                
                # Shape adjustment
                if len(output.shape) == 3:
                    output = output.mean(dim=-1)
                if output.shape[1] > traj_target.shape[1]:
                    output = output[:, :traj_target.shape[1]]
                
                loss = criterion(output * traj_mask, traj_target * traj_mask)
                loss.backward()
                optimizer.step()
        
        print(f"✓ [2/3] 训练完成")
        result['details']['training_completed'] = True
        
        # 3. 推理测试
        print(f"  [3/3] 推理测试...", flush=True)
        model.eval()
        with torch.no_grad():
            sample = next(iter(train_loader))
            x = sample['x'].to(device)
            if x.shape[1] < 3:
                pad_len = 3 - x.shape[1]
                x = torch.cat([x, torch.zeros(x.shape[0], pad_len, x.shape[2], device=device)], dim=1)
            output = model(x)
            print(f"✓ [3/3] 推理完成: output={output.shape}")
            result['details']['inference_completed'] = True
            result['details']['output_shape'] = str(output.shape)
        
        result['status'] = 'success'
        print(f"✅ {name} 测试通过")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"❌ {name} 测试失败: {str(e)}")
        traceback.print_exc()
    
    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    table_path = 'data/landmark_tables/unified_person_landmark_table.pkl'
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=42)
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, landmark_to_idx, batch_size=32)
    
    sample = next(iter(train_loader))
    feature_dim = sample['x'].shape[2]
    
    results = []
    
    # 测试 Landmark Wrappers
    print("\n" + "="*70)
    print("第一部分: Landmark Wrappers")
    print("="*70)
    
    results.append(test_landmark_wrapper('TSDiff (严格迁移)', TSDiffLandmarkWrapper, train_loader, device))
    results.append(test_landmark_wrapper('STaSy (严格迁移)', STaSyLandmarkWrapper, train_loader, device))
    results.append(test_landmark_wrapper('TabSyn (简化 VAE)', TabSynLandmarkWrapper, train_loader, device))
    results.append(test_landmark_wrapper('TabDiff (简化 DDPM)', TabDiffLandmarkWrapper, train_loader, device))
    
    # 测试 TSLib Wrappers
    print("\n" + "="*70)
    print("第二部分: TSLib Wrappers (Layer2)")
    print("="*70)
    
    itransformer = iTransformerWrapper(seq_len=3, enc_in=feature_dim, task='long_term_forecast', pred_len=7)
    results.append(test_tslib_wrapper('iTransformer', itransformer, train_loader, device))
    
    timexer = TimeXerWrapper(seq_len=3, enc_in=feature_dim, exog_in=4, task='long_term_forecast', pred_len=7)
    results.append(test_tslib_wrapper('TimeXer', timexer, train_loader, device))
    
    # 输出结构化结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n总计: {len(results)} 个模型")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {failed_count}")
    print(f"成功率: {success_count/len(results)*100:.1f}%")
    
    print("\n详细结果:")
    for r in results:
        status_icon = "✅" if r['status'] == 'success' else "❌"
        print(f"{status_icon} {r['model']}: {r['status']}")
        if r['error']:
            print(f"   错误: {r['error']}")
    
    # 保存结果
    import json
    with open('outputs/unified_baseline_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: outputs/unified_baseline_test_results.json")


if __name__ == '__main__':
    main()
