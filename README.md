# AIDD8122 期末项目：基于机器学习的分子对接打分函数改进

Improving ML-based scoring functions for protein–ligand binding affinity prediction by combining RF-Score contact features with ligand descriptors.

## 项目一句话

证明「机器学习重打分模型」比传统物理打分函数（AutoDock Vina）更能准确预测蛋白-小分子结合亲和力；并进一步证明在 RF-Score 的 36 维接触特征基础上加入 12 维配体描述子，可将 Pearson 从 0.764 提升到 0.799（在排除训练/评测集重叠、无数据泄漏的严格评测下）。

## 环境

依赖：`rdkit`、`xgboost`、`scikit-learn`、`scipy`、`pandas`、`numpy`、`matplotlib`

```bash
# 建议用 conda 创建独立环境
conda create -n chem_rdkit python=3.11 -y
conda activate chem_rdkit
conda install -c conda-forge rdkit xgboost scikit-learn scipy pandas numpy matplotlib -y
```

## 数据

原始数据**不包含在本仓库中**（体积巨大）。请自行下载并放到如下目录结构：

| 数据 | 路径 | 用途 | 样本数 |
|---|---|---|---|
| PDBbind v2020 refined set | `data set/refined-set/` | 训练 | 5316 |
| CASF-2016 core set | `data set/CASF-2016/` | 评测 | 285 |

下载地址：<http://www.pdbbind.org.cn/download.php>（PDBbind v2020 refined set 与 CASF-2016 core set）。

标签（pKd）来源：
- refined：`data set/refined-set/index/INDEX_refined_data.2020` 第 4 列
- CASF：`data set/CASF-2016/power_scoring/CoreSet.dat` 第 4 列（logKa）

目录结构要求：

```
data set/
├── refined-set/
│   ├── <PDB code>/              # 每个复合物一个子目录，含 *_protein.pdb 和 *_ligand.mol2 / *.sdf
│   └── index/INDEX_refined_data.2020
└── CASF-2016/
    ├── coreset/<PDB code>/      # 285 个评测复合物
    └── power_scoring/CoreSet.dat
```

## 两种运行方式

### 方式 A：Jupyter Notebook（推荐，适合逐步操作/答辩演示）

文件：`final_project.ipynb`

- 用 **VS Code** 打开 `final_project.ipynb`；
- 右上角内核选择 `chem_rdkit`（或 `Python 3.13 (chem_rdkit)`）；
- 从上到下依次点每个 cell 的运行按钮即可。

> Notebook 已把所有脚本整合为 19 个 cell（含说明文字），每步结果会内联显示，方便逐步查看和答辩演示。

### 方式 B：命令行脚本（适合一键复现）

一键运行：`bash run_all.sh`

或手动分步执行下面 5 个脚本。

## 脚本与执行顺序

| 步骤 | 脚本 | 作用 | 状态 |
|---|---|---|---|
| 1 | `src/parse_labels.py` | 解析索引，生成 `results/refined_labels.csv`、`results/casf_labels.csv` | ✅ 已验证 |
| 2 | `src/extract_features.py` | 用 RDKit 算 36 维 RF-Score 接触特征，生成 `*_features.csv` | ✅ 已验证 |
| 3 | `src/baseline_rfscore.py` | RF-Score 基线（随机森林），生成 `baseline_predictions.csv` | ✅ 已验证 |
| 4 | `src/improved_model.py` | 改进模型（XGBoost/RF + 配体描述子 + 消融 + 无泄漏），生成 `comparison.csv`、`ablation.csv` | ✅ 已验证 |
| 5 | `src/plot_results.py` | 画散点图 + 对比柱状图 + 消融图 | ✅ 已验证 |

### 一键跑通全流程（按顺序执行）：

```bash
# 用 chem_rdkit 环境的 python（项目根目录下执行）
PY="C:/Users/xinji/anaconda3/envs/chem_rdkit/python.exe"
"$PY" src/parse_labels.py
"$PY" src/extract_features.py
"$PY" src/baseline_rfscore.py
"$PY" src/improved_model.py
"$PY" src/plot_results.py
```

> 注：第 2 步特征提取需几分钟（5316 个复合物），其余步骤均很快。

## 方法说明

### 特征（RF-Score 式，36 维）
- 将蛋白原子、配体原子各按元素分为 6 类：C, N, O, S, 卤素(X), 其他
- 对 6×6=36 种「蛋白类型 × 配体类型」组合，统计 12 Å 距离内的接触原子对数量
- 用 RDKit 解析蛋白 PDB 和配体 mol2/sdf，计算欧氏距离

### 基线
- **RF-Score**：随机森林（n_estimators=500）在 36 维接触特征上回归 pKd

### 改进
- 在 36 维接触特征基础上，**加入 12 维配体 RDKit 描述子**（分子量、LogP、氢键受体/供体、可旋转键、TPSA 等）
- 对比 4 个模型变体（RF / XGBoost × 有/无描述子），并做消融（只用接触特征 / 只用描述子 / 两者结合，均用随机森林）

### 无数据泄漏保证
- CASF-2016 core set 原本从 PDBbind refined set 中抽取，两者存在 266 个重叠复合物（占评测集 93%）
- 训练前已**剔除这 266 个重叠样本**（最终训练集 5,050 个），保证评测无数据泄漏

### 评估指标
- RMSE（均方根误差）
- Pearson 相关系数（scoring power）
- Spearman 秩相关系数（ranking power）

## 已得到的结果（最终，无泄漏，可写进报告）

| 模型 | Pearson | Spearman | RMSE |
|---|---|---|---|
| RF-Score（随机森林，36 接触特征）基线 | 0.764 | 0.759 | 1.484 |
| XGBoost（36 接触特征） | 0.750 | 0.744 | 1.476 |
| XGBoost（接触+描述子） | 0.798 | 0.786 | 1.374 |
| **RF（接触+描述子）← 最优** | **0.799** | **0.792** | **1.411** |

**消融实验（随机森林）**：

| 特征组合 | Pearson | Spearman | RMSE |
|---|---|---|---|
| 仅接触特征（36 维） | 0.764 | 0.759 | 1.484 |
| 仅描述子（12 维） | 0.658 | 0.640 | 1.659 |
| **两者结合（48 维）** | **0.799** | **0.792** | **1.411** |

> 核心结论：在 36 维接触特征基础上加入 12 维配体 RDKit 描述子后，
> 随机森林模型的 Pearson 从 0.764 提升到 0.799，RMSE 从 1.484 降到 1.411，
> 证明了「接触特征 + 配体描述子」的互补性（消融实验也验证了这一点）。
>
> ⚠️ 说明：这些结果是在**排除训练/评测集重叠后**得到的（无数据泄漏的严格评测）。
> 若不做泄漏剔除，Pearson 会被高估到约 0.939；本仓库所有结果均为剔除后的真实值。

## 可能遇到的问题

1. **XGBoost 仍略低于 RF-Score**：正常。可尝试：
   - 调 XGBoost 超参（max_depth 4~10、learning_rate 0.01~0.1）
   - 增加更多描述子（rdMolDescriptors 里有 200+ 个）
   - 换 LightGBM / 简单 MLP
2. **特征提取报错**：已在 `extract_features.py` 做了元素解析容错 + 单样本异常保护，单个复合物失败会跳过并在末尾打印。
3. **为什么结果和 RF-Score 原论文不同**：因为本仓库在训练前**剔除了与 CASF-2016 重叠的 266 个复合物**（无泄漏评测），所以 Pearson 约为 0.76–0.80；若沿用 CASF 传统的"允许训练集包含 core set"协议，Pearson 会更高（约 0.94），但那包含了数据泄漏。

## 项目结构

```
.
├── src/
│   ├── parse_labels.py       # 1. 解析标签 → refined_labels.csv / casf_labels.csv
│   ├── extract_features.py   # 2. 36 维接触特征提取（RDKit）
│   ├── baseline_rfscore.py   # 3. RF-Score 基线
│   ├── improved_model.py     # 4. 改进模型（+12 描述子、XGBoost/RF、消融、无泄漏剔除）
│   └── plot_results.py       # 5. 可视化（散点图 + 对比柱状图 + 消融图）
├── final_project.ipynb       # 全部流程整合为 Notebook（推荐逐步操作）
├── run_all.sh                # 一键复现脚本
└── results/                  # 输出（特征 CSV、预测、对比表、图）
```

## 引用

若本项目对你有帮助，可参考以下相关工作：

- Ballester, P. J., & Mitchell, J. B. O. (2010). A machine learning approach to predicting protein–ligand binding affinity with applications to molecular docking. *Bioinformatics*, 26(9), 1169–1175.
- Su, M., et al. (2019). Comparative assessment of scoring functions: The CASF-2016 update. *Journal of Chemical Information and Modeling*, 59(2), 895–913.
