"""JBHI 吸烟条件化复现实验的固定协议工具。

这个模块刻意把论文所需的关键条件集中在一处：
``baseline_cigsmok`` 必须从普通输入中移除，只能作为 alpha 条件；
14 维输入、运行目录和元数据也必须被明确记录。旧的 landmark 运行
仍可保留，但不能借用本模块的协议名称。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


JBHI_PROTOCOL = 'jbhi_smoking'
LEGACY_PROTOCOL = 'legacy_landmark'
SMOKING_TREATMENT_COLUMN = 'baseline_cigsmok'
EXPECTED_FEATURE_DIM = 14


def activate_jbhi_smoking_condition() -> str:
    """固定真实条件列，拒绝与旧环境变量混用。"""
    existing = os.environ.get('ALPHA_TREATMENT_COLUMN')
    if existing and existing != SMOKING_TREATMENT_COLUMN:
        raise RuntimeError(
            'JBHI 吸烟条件协议只允许 '
            f'ALPHA_TREATMENT_COLUMN={SMOKING_TREATMENT_COLUMN}，'
            f'当前却是 {existing!r}。'
        )
    os.environ['ALPHA_TREATMENT_COLUMN'] = SMOKING_TREATMENT_COLUMN
    return SMOKING_TREATMENT_COLUMN


def jbhi_run_paths(seed: int, run_dir: Optional[str] = None) -> Dict[str, Path]:
    """返回互不覆盖的单 seed 产物路径。"""
    root = Path(run_dir) if run_dir else Path('outputs') / 'jbhi_smoking' / f'seed{seed}'
    return {
        'root': root,
        'log_dir': root / 'logs',
        'phase2_log': root / 'logs' / 'phase2.log',
        'checkpoint': root / 'checkpoints' / 'phase2_best.pt',
        'prediction': root / 'predictions' / 'phase2_predictions.npz',
        'efficiency': root / 'efficiency.json',
        'manifest': root / 'manifest.json',
        'validation': root / 'causal_validation.json',
        'logistic_log': root / 'logs' / 'logistic_baseline.log',
        'logistic_prediction': root / 'logistic_baseline.npz',
        'logistic_model': root / 'logistic_baseline.pkl',
    }


def prepare_jbhi_run_directory(paths: Dict[str, Path]) -> None:
    """创建不可变的新运行目录，拒绝残留文件与覆盖运行。"""
    root = paths['root']
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f'运行目录并非空目录：{root}。'
            '为避免新旧 checkpoint、日志或预测混用，请换一个新的 --run_dir。'
        )
    for key in (
        'root', 'log_dir', 'checkpoint', 'prediction', 'efficiency', 'validation',
        'logistic_log', 'logistic_prediction', 'logistic_model',
    ):
        path = paths[key]
        (path if key in ('root', 'log_dir') else path.parent).mkdir(parents=True, exist_ok=True)


def sha256_file(path: str | Path) -> str:
    """流式计算数据表哈希，避免把整个 pickle 放入内存。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(cwd: str | Path = '.') -> Dict[str, Any]:
    """记录代码版本；没有 Git 元数据的快照也能安全降级。"""
    cwd = str(cwd)

    def _git(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ['git', *args], cwd=cwd, check=False, capture_output=True, text=True
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain')
    return {
        'git_commit': commit,
        'git_dirty': None if status is None else bool(status),
    }


def critical_source_fingerprint(cwd: str | Path) -> Dict[str, Any]:
    """记录本复现链关键源码的逐文件与组合哈希。

    服务器工作树可能带有未提交修订；仅记录 ``git_dirty=true`` 无法以后还原
    实际代码。因此只对训练、诊断、数据模块和核心模型的有限集合做可读哈希，
    既避免扫描大数据目录，也能锁定这条复现链的关键实现。
    """
    root = Path(cwd)
    relative_paths = (
        'run_phase2_twostage.py',
        'run_causal_validation.py',
        'aggregate_causal_validation.py',
        'scripts/run_jbhi_logistic_baseline.py',
        'scripts/preflight_jbhi_smoking.sh',
        'scripts/run_jbhi_smoking_seed.sh',
        'scripts/run_jbhi_smoking_5seeds.sh',
        'src/utils/jbhi_protocol.py',
        'src/data/data_module_landmark.py',
        'src/data/data_module_landmark_cached.py',
        'src/models/causal_tabdiff.py',
        'src/models/causal_tabdiff_trajectory.py',
        'src/evaluation/metrics.py',
        'src/evaluation/efficiency.py',
        'src/utils/training_logger.py',
    )
    files: Dict[str, Optional[str]] = {}
    for relative_path in relative_paths:
        path = root / relative_path
        files[relative_path] = sha256_file(path) if path.is_file() else None
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {
        'critical_files': files,
        'combined_sha256': hashlib.sha256(encoded).hexdigest(),
    }


def write_manifest(path: str | Path, payload: Dict[str, Any]) -> None:
    """原子写入 manifest，避免中断时留下半个 JSON 证据文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent, prefix=f'.{path.name}.', delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
        temporary_path = Path(handle.name)
    temporary_path.replace(path)
