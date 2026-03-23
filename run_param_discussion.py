"""
参数讨论实验 (Parameter Discussion Experiments)
======================================================
4个关键参数 × 5个值 × 5 seeds = 100 组实验
参数:
  1. disc_weight:     [0.0, 0.25, 0.5, 1.0, 2.0]   — 因果判别器权重
  2. guidance_scale:  [0.0, 0.5, 1.0, 2.0, 5.0]     — 推理期因果梯度引导强度
  3. diffusion_steps: [25, 50, 100, 200, 500]         — 扩散步数
  4. learning_rate:   [1e-4, 3e-4, 5e-4, 1e-3, 3e-3] — 学习率

评估方式:
  - disc_weight / diffusion_steps / learning_rate: Direct Forward AUPRC
  - guidance_scale: 训练1次 + 5个scale采样评估 Sampled AUPRC

遵循 AGENTS.md 训练日志规范
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_module_landmark import load_and_split_data, create_dataloaders
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
from src.evaluation.metrics import compute_ranking_metrics

# ======================== 全局配置 ========================
SEEDS = [42, 52, 62, 72, 82]
EPOCHS = 30
BATCH_SIZE = 4096
DEFAULT_TABLE_PATH = 'data/landmark_tables/unified_person_landmark_table.pkl'
OUTPUT_DIR = 'outputs/param_discussion'

# 默认超参数 (控制变量法: 变一个参数时，其余固定为 default)
DEFAULTS = {
    'disc_weight': 1.0,
    'traj_weight': 1.0,
    'diffusion_steps': 100,
    'learning_rate': 5e-4,
    'guidance_scale': 2.0,
}

# 参数讨论配置
PARAM_CONFIGS = {
    'disc_weight': {
        'name': '因果判别器权重 (disc_weight)',
        'values': [0.0, 0.25, 0.5, 1.0, 2.0],
        'eval_mode': 'direct',
        'description': '控制因果判别器损失在训练目标中的权重，反映因果约束的强度',
    },
    'guidance_scale': {
        'name': '因果梯度引导强度 (guidance_scale)',
        'values': [0.0, 0.5, 1.0, 2.0, 5.0],
        'eval_mode': 'sampled',
        'description': '推理阶段因果梯度引导的强度，控制生成样本的因果一致性',
    },
    'diffusion_steps': {
        'name': '扩散步数 (diffusion_steps)',
        'values': [25, 50, 100, 200, 500],
        'eval_mode': 'direct',
        'description': '前向加噪/反向去噪步数，影响分布建模精细度和计算开销',
    },
    'learning_rate': {
        'name': '学习率 (learning_rate)',
        'values': [1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
        'eval_mode': 'direct',
        'description': 'Adam 优化器学习率，影响训练收敛速度和稳定性',
    },
}


def log_and_print(msg, log_fh=None):
    """同时输出到终端和日志文件"""
    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + '\n')
        log_fh.flush()


def build_alpha_target(batch, device):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    if 'landmark' in batch:
        return batch['landmark'].float().to(device)
    raise KeyError("Batch must contain 'alpha_target' or 'landmark'")


def train_and_evaluate_direct(seed, disc_weight, traj_weight, diffusion_steps, lr, epochs, log_fh=None):
    """训练模型并用 Direct Forward 评估，返回 best_val_auprc, best_val_auroc, final_test_auprc, final_test_auroc"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 数据加载
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(DEFAULT_TABLE_PATH, seed=seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=BATCH_SIZE, num_workers=4
    )

    # 模型
    model = CausalTabDiffTrajectory(
        t_steps=3, feature_dim=15, diffusion_steps=diffusion_steps, trajectory_len=7,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auprc = 0.0
    best_val_auroc = 0.0
    best_state = None

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            y_2year = batch['y_2year'].to(device)
            traj_target = batch['trajectory_target'].to(device)
            traj_mask = batch['trajectory_valid_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(x, alpha)

            loss = (outputs['diff_loss']
                    + disc_weight * outputs['disc_loss']
                    + traj_weight * model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask)
                    + F.binary_cross_entropy(outputs['risk_2year'], y_2year))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1

        train_loss = total_train_loss / max(n_batches, 1)

        # 验证
        model.eval()
        all_val_y_true, all_val_y_pred = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                alpha = build_alpha_target(batch, device)
                y_2year = batch['y_2year'].to(device)
                traj_target = batch['trajectory_target'].to(device)
                traj_mask = batch['trajectory_valid_mask'].to(device)

                outputs = model(x, alpha)
                vloss = (outputs['diff_loss']
                         + disc_weight * outputs['disc_loss']
                         + traj_weight * model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask)
                         + F.binary_cross_entropy(outputs['risk_2year'], y_2year))
                val_loss_total += vloss.item()
                all_val_y_true.append(y_2year.cpu().numpy())
                all_val_y_pred.append(outputs['risk_2year'].cpu().numpy())

        val_loss = val_loss_total / max(len(val_loader), 1)
        val_y_true = np.concatenate(all_val_y_true).flatten()
        val_y_pred = np.concatenate(all_val_y_pred).flatten()
        val_metrics = compute_ranking_metrics(val_y_true, val_y_pred)
        val_auprc = val_metrics['auprc'] if not np.isnan(val_metrics['auprc']) else 0.0
        val_auroc = val_metrics['auroc'] if not np.isnan(val_metrics['auroc']) else 0.0

        elapsed = time.time() - t0

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 日志输出
        log_msg = (f"Epoch {epoch+1}/{epochs} | Seed {seed} | LR {lr:.1e} | "
                   f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
                   f"ValAUPRC {val_auprc:.4f} | ValAUROC {val_auroc:.4f} | "
                   f"BestValAUPRC {best_val_auprc:.4f} | Time {elapsed:.1f}s")
        log_and_print(log_msg, log_fh)

    # 用最佳模型在测试集上评估
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    all_test_y_true, all_test_y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            y_2year = batch['y_2year'].to(device)
            outputs = model(x, alpha)
            all_test_y_true.append(y_2year.cpu().numpy())
            all_test_y_pred.append(outputs['risk_2year'].cpu().numpy())

    test_y_true = np.concatenate(all_test_y_true).flatten()
    test_y_pred = np.concatenate(all_test_y_pred).flatten()
    test_metrics = compute_ranking_metrics(test_y_true, test_y_pred)
    test_auprc = test_metrics['auprc'] if not np.isnan(test_metrics['auprc']) else 0.0
    test_auroc = test_metrics['auroc'] if not np.isnan(test_metrics['auroc']) else 0.0

    return {
        'best_val_auprc': best_val_auprc,
        'best_val_auroc': best_val_auroc,
        'test_auprc': test_auprc,
        'test_auroc': test_auroc,
    }


def train_and_evaluate_sampled(seed, guidance_values, disc_weight, traj_weight, diffusion_steps, lr, epochs, log_fh=None):
    """
    训练1次模型，然后用不同 guidance_scale 采样评估。
    返回 {guidance_scale: {val_auprc, val_auroc, test_auprc, test_auroc}}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(DEFAULT_TABLE_PATH, seed=seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=BATCH_SIZE, num_workers=4
    )

    model = CausalTabDiffTrajectory(
        t_steps=3, feature_dim=15, diffusion_steps=diffusion_steps, trajectory_len=7,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auprc = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch['x'].to(device)
            alpha = build_alpha_target(batch, device)
            y_2year = batch['y_2year'].to(device)
            traj_target = batch['trajectory_target'].to(device)
            traj_mask = batch['trajectory_valid_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(x, alpha)
            loss = (outputs['diff_loss']
                    + disc_weight * outputs['disc_loss']
                    + traj_weight * model.compute_trajectory_loss(outputs['trajectory'], traj_target, traj_mask)
                    + F.binary_cross_entropy(outputs['risk_2year'], y_2year))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1

        train_loss = total_train_loss / max(n_batches, 1)

        # 简单验证 (Direct Forward)
        model.eval()
        all_vy, all_vp = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                alpha = build_alpha_target(batch, device)
                y_2year = batch['y_2year'].to(device)
                outputs = model(x, alpha)
                all_vy.append(y_2year.cpu().numpy())
                all_vp.append(outputs['risk_2year'].cpu().numpy())
        vy = np.concatenate(all_vy).flatten()
        vp = np.concatenate(all_vp).flatten()
        vm = compute_ranking_metrics(vy, vp)
        vauprc = vm['auprc'] if not np.isnan(vm['auprc']) else 0.0
        if vauprc > best_val_auprc:
            best_val_auprc = vauprc

        elapsed = time.time() - t0
        log_msg = (f"Epoch {epoch+1}/{epochs} | Seed {seed} | LR {lr:.1e} | "
                   f"TrainLoss {train_loss:.4f} | ValAUPRC(direct) {vauprc:.4f} | "
                   f"BestValAUPRC {best_val_auprc:.4f} | Time {elapsed:.1f}s")
        log_and_print(log_msg, log_fh)

    # 对每个 guidance_scale 进行采样评估
    results_by_scale = {}
    model.eval()

    for g_scale in guidance_values:
        log_and_print(f"  采样评估 | guidance_scale={g_scale} | Seed={seed}", log_fh)

        # 验证集采样评估
        all_val_y_true, all_val_y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                alpha = build_alpha_target(batch, device)
                y_2year = batch['y_2year'].to(device)
                bs = alpha.shape[0]

                x_gen = model.sample_with_guidance(bs, alpha, guidance_scale=g_scale)
                # 通过 block1 + trajectory_head -> risk_2year
                c_emb = model.base_model.cond_mlp(alpha)
                h = model.base_model.block1(x_gen, c_emb)
                h_pooled = h.mean(dim=1)
                traj_logits = model.trajectory_head(h_pooled)
                traj_probs = torch.sigmoid(traj_logits)
                risk_2year = model.compute_2year_risk(traj_probs)

                all_val_y_true.append(y_2year.cpu().numpy())
                all_val_y_pred.append(risk_2year.cpu().numpy())

        val_y_true = np.concatenate(all_val_y_true).flatten()
        val_y_pred = np.concatenate(all_val_y_pred).flatten()
        val_m = compute_ranking_metrics(val_y_true, val_y_pred)

        # 测试集采样评估
        all_test_y_true, all_test_y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                alpha = build_alpha_target(batch, device)
                y_2year = batch['y_2year'].to(device)
                bs = alpha.shape[0]

                x_gen = model.sample_with_guidance(bs, alpha, guidance_scale=g_scale)
                c_emb = model.base_model.cond_mlp(alpha)
                h = model.base_model.block1(x_gen, c_emb)
                h_pooled = h.mean(dim=1)
                traj_logits = model.trajectory_head(h_pooled)
                traj_probs = torch.sigmoid(traj_logits)
                risk_2year = model.compute_2year_risk(traj_probs)

                all_test_y_true.append(y_2year.cpu().numpy())
                all_test_y_pred.append(risk_2year.cpu().numpy())

        test_y_true = np.concatenate(all_test_y_true).flatten()
        test_y_pred = np.concatenate(all_test_y_pred).flatten()
        test_m = compute_ranking_metrics(test_y_true, test_y_pred)

        results_by_scale[str(g_scale)] = {
            'val_auprc': float(val_m['auprc']) if not np.isnan(val_m['auprc']) else 0.0,
            'val_auroc': float(val_m['auroc']) if not np.isnan(val_m['auroc']) else 0.0,
            'test_auprc': float(test_m['auprc']) if not np.isnan(test_m['auprc']) else 0.0,
            'test_auroc': float(test_m['auroc']) if not np.isnan(test_m['auroc']) else 0.0,
        }

        log_and_print(
            f"    guidance_scale={g_scale} | ValAUPRC(sampled) {results_by_scale[str(g_scale)]['val_auprc']:.4f} | "
            f"TestAUPRC(sampled) {results_by_scale[str(g_scale)]['test_auprc']:.4f}", log_fh)

    return results_by_scale


def run_param_experiment(param_name):
    """运行单个参数的讨论实验"""
    config = PARAM_CONFIGS[param_name]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('logs/param_discussion', exist_ok=True)

    log_path = f'logs/param_discussion/{param_name}.log'
    log_fh = open(log_path, 'w', buffering=1)

    log_and_print(f"{'='*70}", log_fh)
    log_and_print(f"参数讨论实验: {config['name']}", log_fh)
    log_and_print(f"描述: {config['description']}", log_fh)
    log_and_print(f"测试值: {config['values']}", log_fh)
    log_and_print(f"评估方式: {config['eval_mode']}", log_fh)
    log_and_print(f"Seeds: {SEEDS}", log_fh)
    log_and_print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}", log_fh)
    log_and_print(f"默认配置: {DEFAULTS}", log_fh)
    log_and_print(f"{'='*70}\n", log_fh)

    all_results = {}

    if config['eval_mode'] == 'sampled':
        # guidance_scale 特殊处理: 每个seed训练1次，5个scale采样
        for seed in SEEDS:
            log_and_print(f"\n--- Seed={seed} 训练开始 ---", log_fh)
            seed_results = train_and_evaluate_sampled(
                seed=seed,
                guidance_values=config['values'],
                disc_weight=DEFAULTS['disc_weight'],
                traj_weight=DEFAULTS['traj_weight'],
                diffusion_steps=DEFAULTS['diffusion_steps'],
                lr=DEFAULTS['learning_rate'],
                epochs=EPOCHS,
                log_fh=log_fh,
            )
            # 转置: {scale: {metrics}} -> {scale: [per-seed metrics]}
            for scale_str, metrics in seed_results.items():
                if scale_str not in all_results:
                    all_results[scale_str] = []
                all_results[scale_str].append(metrics)

            log_and_print(f"--- Seed={seed} 完成 ---\n", log_fh)

    else:
        # disc_weight / diffusion_steps / learning_rate: 逐值逐seed训练+评估
        for value in config['values']:
            value_str = str(value)
            all_results[value_str] = []

            for seed in SEEDS:
                log_and_print(f"\n--- {param_name}={value} | Seed={seed} 训练开始 ---", log_fh)

                # 组装超参: 控制变量法
                cur_disc_weight = DEFAULTS['disc_weight']
                cur_traj_weight = DEFAULTS['traj_weight']
                cur_diffusion_steps = DEFAULTS['diffusion_steps']
                cur_lr = DEFAULTS['learning_rate']

                if param_name == 'disc_weight':
                    cur_disc_weight = value
                elif param_name == 'diffusion_steps':
                    cur_diffusion_steps = value
                elif param_name == 'learning_rate':
                    cur_lr = value

                result = train_and_evaluate_direct(
                    seed=seed,
                    disc_weight=cur_disc_weight,
                    traj_weight=cur_traj_weight,
                    diffusion_steps=cur_diffusion_steps,
                    lr=cur_lr,
                    epochs=EPOCHS,
                    log_fh=log_fh,
                )

                all_results[value_str].append(result)
                log_and_print(
                    f"  结果 | {param_name}={value} | Seed={seed} | "
                    f"BestValAUPRC={result['best_val_auprc']:.4f} | "
                    f"TestAUPRC={result['test_auprc']:.4f}", log_fh)

    # ======================== 汇总 ========================
    log_and_print(f"\n{'='*70}", log_fh)
    log_and_print(f"{'='*20} 汇总: {config['name']} {'='*20}", log_fh)
    log_and_print(f"{'='*70}\n", log_fh)

    summary = {}
    for value in config['values']:
        value_str = str(value)
        entries = all_results[value_str]

        if config['eval_mode'] == 'sampled':
            val_auprcs = [e['val_auprc'] for e in entries]
            test_auprcs = [e['test_auprc'] for e in entries]
        else:
            val_auprcs = [e['best_val_auprc'] for e in entries]
            test_auprcs = [e['test_auprc'] for e in entries]

        summary[value_str] = {
            'val_auprc_mean': float(np.mean(val_auprcs)),
            'val_auprc_std': float(np.std(val_auprcs)),
            'test_auprc_mean': float(np.mean(test_auprcs)),
            'test_auprc_std': float(np.std(test_auprcs)),
            'val_auprcs': val_auprcs,
            'test_auprcs': test_auprcs,
        }

        log_and_print(
            f"{param_name}={value:>8} | "
            f"ValAUPRC {summary[value_str]['val_auprc_mean']:.4f} ± {summary[value_str]['val_auprc_std']:.4f} | "
            f"TestAUPRC {summary[value_str]['test_auprc_mean']:.4f} ± {summary[value_str]['test_auprc_std']:.4f}",
            log_fh)

    # 保存
    output = {
        'param_name': param_name,
        'config': {k: v for k, v in config.items() if k != 'values'},
        'values': [str(v) for v in config['values']],
        'defaults': DEFAULTS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'seeds': SEEDS,
        'raw_results': all_results,
        'summary': summary,
    }

    json_path = os.path.join(OUTPUT_DIR, f'param_discussion_{param_name}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    log_and_print(f"\n结果已保存: {json_path}", log_fh)
    log_and_print(f"日志已保存: {log_path}", log_fh)
    log_fh.close()

    return summary


def run_all():
    """串行运行所有4个参数实验"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('logs/param_discussion', exist_ok=True)

    master_log_path = 'logs/param_discussion/master.log'
    master_fh = open(master_log_path, 'w', buffering=1)

    log_and_print(f"{'='*70}", master_fh)
    log_and_print(f"Causal-TabDiff 参数讨论实验 (Parameter Discussion)", master_fh)
    log_and_print(f"4 参数 × 5 值 × 5 seeds = 100 组", master_fh)
    log_and_print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", master_fh)
    log_and_print(f"{'='*70}\n", master_fh)

    all_summaries = {}
    param_order = ['disc_weight', 'diffusion_steps', 'learning_rate', 'guidance_scale']

    for i, param_name in enumerate(param_order):
        t0 = time.time()
        log_and_print(f"\n[{i+1}/4] 开始: {PARAM_CONFIGS[param_name]['name']}", master_fh)

        summary = run_param_experiment(param_name)
        all_summaries[param_name] = summary

        elapsed = time.time() - t0
        log_and_print(f"[{i+1}/4] 完成: {PARAM_CONFIGS[param_name]['name']} | 耗时 {elapsed/60:.1f} 分钟", master_fh)

    # 总汇总
    log_and_print(f"\n{'='*70}", master_fh)
    log_and_print(f"{'='*20} 全局汇总 {'='*20}", master_fh)
    log_and_print(f"{'='*70}\n", master_fh)

    for param_name in param_order:
        config = PARAM_CONFIGS[param_name]
        summary = all_summaries[param_name]
        log_and_print(f"\n--- {config['name']} ---", master_fh)
        for value in config['values']:
            s = summary[str(value)]
            log_and_print(
                f"  {param_name}={value:>8} | "
                f"ValAUPRC {s['val_auprc_mean']:.4f}±{s['val_auprc_std']:.4f} | "
                f"TestAUPRC {s['test_auprc_mean']:.4f}±{s['test_auprc_std']:.4f}",
                master_fh)

    # 保存总结果
    master_json = os.path.join(OUTPUT_DIR, 'param_discussion_all.json')
    with open(master_json, 'w') as f:
        json.dump(all_summaries, f, indent=2, default=str)

    log_and_print(f"\n全局结果已保存: {master_json}", master_fh)
    log_and_print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", master_fh)
    master_fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal-TabDiff 参数讨论实验')
    parser.add_argument('--param', type=str, default='all',
                        choices=['all'] + list(PARAM_CONFIGS.keys()),
                        help='运行哪个参数实验 (default: all)')
    args = parser.parse_args()

    if args.param == 'all':
        run_all()
    else:
        run_param_experiment(args.param)
