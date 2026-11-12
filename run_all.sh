#!/usr/bin/env bash
# =============================================================================
# run_all.sh —— 一键复现整套流程
# 用法：在项目根目录下执行  bash run_all.sh
# 作用：按顺序跑完 5 个脚本，输出结果到 results/ 目录
# =============================================================================

set -e  # 任一步失败立即停止

# Python 解释器：默认使用系统 python；若用 conda 环境，可这样指定：
#   PY="/path/to/envs/chem_rdkit/python" bash run_all.sh
# 依赖：rdkit, xgboost, scikit-learn, scipy, pandas, numpy, matplotlib
PY="${PY:-python}"

echo "=============================================="
echo " 步骤 1/5：解析标签 (parse_labels.py)"
echo "=============================================="
"$PY" src/parse_labels.py

echo ""
echo "=============================================="
echo " 步骤 2/5：提取 36 维接触特征 (extract_features.py)"
echo "=============================================="
echo "  （约 1.5 分钟，5316 个复合物，请耐心等待）"
"$PY" src/extract_features.py 2>&1 | grep -vE "Warning|hydrogens|valence|kekulize|sanitize|amidine"

echo ""
echo "=============================================="
echo " 步骤 3/5：RF-Score 基线 (baseline_rfscore.py)"
echo "=============================================="
"$PY" src/baseline_rfscore.py

echo ""
echo "=============================================="
echo " 步骤 4/5：改进模型 + 消融 (improved_model.py)"
echo "=============================================="
"$PY" src/improved_model.py

echo ""
echo "=============================================="
echo " 步骤 5/5：画图 (plot_results.py)"
echo "=============================================="
"$PY" src/plot_results.py

echo ""
echo "=============================================="
echo " ✅ 全部完成！结果在 results/ 目录："
echo "    - comparison.csv              模型对比表"
echo "    - improved_predictions.csv    最优模型预测"
echo "    - figures/                    两张图"
echo "=============================================="
