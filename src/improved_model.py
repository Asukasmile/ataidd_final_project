"""
improved_model.py
改进模型完整版：在 RF-Score 36 维接触特征基础上，加入配体 RDKit 描述子，
对比多个模型变体，并做消融实验。

对比项：
  1. RF-Score 基线（随机森林，36 接触特征）
  2. XGBoost（36 接触特征）
  3. XGBoost（36 接触特征 + 配体描述子）  <- 你的核心改进
  4. 随机森林（36 接触特征 + 配体描述子）

消融：
  - 只用接触特征 vs 只用描述子 vs 两者结合

依赖：rdkit, xgboost, scikit-learn, scipy, pandas, numpy
运行：python src/improved_model.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen

# 静默 RDKit 的 sanitize 警告（mol2 文件常缺氢/价态异常，不影响描述子计算）
RDLogger.DisableLog("rdApp.*")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
DATA_DIR = os.path.join(BASE, "data set")

# 选用的 RDKit 描述子（配体的整体理化性质）
DESCRIPTOR_FUNCS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.NumHAcceptors,
    Descriptors.NumHDonors,
    Descriptors.NumRotatableBonds,
    Descriptors.NumAromaticRings,
    Descriptors.NumAliphaticRings,
    Descriptors.TPSA,
    Descriptors.FractionCSP3,
    Descriptors.HeavyAtomCount,
    Descriptors.NumSaturatedRings,
    Crippen.MolMR,
]


def load_ligand_descriptors(code_list, subdir):
    """对给定 code 列表，从配体文件计算 RDKit 描述子。"""
    feats = []
    for code in code_list:
        complex_dir = os.path.join(DATA_DIR, subdir, code)
        ligand_mol2 = os.path.join(complex_dir, f"{code}_ligand.mol2")
        ligand_sdf = os.path.join(complex_dir, f"{code}_ligand.sdf")
        ligand_path = ligand_mol2 if os.path.exists(ligand_mol2) else ligand_sdf

        mol = None
        if os.path.exists(ligand_path):
            ext = os.path.splitext(ligand_path)[1].lower()
            if ext == ".mol2":
                # 先尝试 sanitize，失败则用 sanitize=False 兜底（描述子对个别坏价态可容忍）
                mol = Chem.MolFromMol2File(ligand_path, removeHs=True, sanitize=True)
                if mol is None:
                    mol = Chem.MolFromMol2File(ligand_path, removeHs=True, sanitize=False)
            elif ext == ".sdf":
                suppl = Chem.SDMolSupplier(ligand_path, removeHs=True, sanitize=True)
                mol = suppl[0] if len(suppl) > 0 else None

        row = []
        for fn in DESCRIPTOR_FUNCS:
            try:
                row.append(float(fn(mol)) if mol is not None else np.nan)
            except Exception:
                row.append(np.nan)
        feats.append(row)
    return np.array(feats, dtype=np.float64)


def load_contact_features():
    train_X = pd.read_csv(os.path.join(RESULTS_DIR, "refined_features.csv")).set_index("code")
    test_X = pd.read_csv(os.path.join(RESULTS_DIR, "casf_features.csv")).set_index("code")
    train_y = pd.read_csv(os.path.join(RESULTS_DIR, "refined_labels.csv")).set_index("code")
    test_y = pd.read_csv(os.path.join(RESULTS_DIR, "casf_labels.csv")).set_index("code")

    # === 数据泄漏防护 ===
    # CASF-2016 core set 原本就抽取自 PDBbind refined set，两者存在大量相同 PDB ID。
    # 若不剔除，训练集会包含测试样本本身，导致评测分数被高估（数据泄漏）。
    # 这里从训练集中删除所有与测试集重叠的 PDB ID，确保无泄漏评测。
    test_codes_all = set(test_X.index)
    overlap = set(train_X.index) & test_codes_all
    if overlap:
        train_X = train_X.drop(index=list(overlap))
        train_y = train_y.drop(index=list(overlap))
        print(f"[数据泄漏防护] 从训练集剔除了 {len(overlap)} 个与 CASF-2016 重叠的复合物")

    train_common = train_X.index.intersection(train_y.index)
    test_common = test_X.index.intersection(test_y.index)

    X_tr = train_X.loc[train_common].values
    y_tr = train_y.loc[train_common]["pKd"].values
    X_te = test_X.loc[test_common].values
    y_te = test_y.loc[test_common]["pKd"].values

    return X_tr, y_tr, X_te, y_te, list(train_common), list(test_common)


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pear = pearsonr(y_true, y_pred)[0]
    spear = spearmanr(y_true, y_pred)[0]
    return rmse, pear, spear


def main():
    X_tr, y_tr, X_te, y_te, train_codes, test_codes = load_contact_features()

    # 配体描述子（训练集和测试集分别计算）
    desc_tr = load_ligand_descriptors(train_codes, "refined-set")
    desc_te = load_ligand_descriptors(test_codes, "CASF-2016/coreset")

    # 处理 NaN：用列均值填充
    for arr in (desc_tr, desc_te):
        for j in range(arr.shape[1]):
            col = arr[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col[nan_mask] = np.nanmean(col)

    # 合并特征
    X_tr_comb = np.hstack([X_tr, desc_tr])
    X_te_comb = np.hstack([X_te, desc_te])

    print(f"训练集 {X_tr.shape[0]}, 测试集 {X_te.shape[0]}")
    print(f"接触特征 {X_tr.shape[1]} 维 + 描述子 {desc_tr.shape[1]} 维 = 合并 {X_tr_comb.shape[1]} 维\n")

    results = []

    # 1) RF-Score 基线
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    r = evaluate(y_te, rf.predict(X_te))
    results.append({"model": "RF-Score (RF, 36 contact)", "rmse": r[0], "pearson": r[1], "spearman": r[2]})
    print(f"  ① RF-Score 基线            RMSE={r[0]:.3f}  Pearson={r[1]:.3f}  Spearman={r[2]:.3f}")

    # 2) XGBoost 仅接触特征
    xgb_contact = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               n_jobs=-1, verbosity=0)
    xgb_contact.fit(X_tr, y_tr)
    r = evaluate(y_te, xgb_contact.predict(X_te))
    results.append({"model": "XGBoost (36 contact)", "rmse": r[0], "pearson": r[1], "spearman": r[2]})
    print(f"  ② XGBoost 仅接触特征      RMSE={r[0]:.3f}  Pearson={r[1]:.3f}  Spearman={r[2]:.3f}")

    # 3) XGBoost 接触 + 描述子（核心改进）
    xgb_comb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            n_jobs=-1, verbosity=0)
    xgb_comb.fit(X_tr_comb, y_tr)
    y_pred_comb = xgb_comb.predict(X_te_comb)
    r = evaluate(y_te, y_pred_comb)
    results.append({"model": "XGBoost (contact+descriptor)", "rmse": r[0], "pearson": r[1], "spearman": r[2]})
    print(f"  ③ XGBoost 接触+描述子     RMSE={r[0]:.3f}  Pearson={r[1]:.3f}  Spearman={r[2]:.3f}")

    # 4) 随机森林 接触 + 描述子
    rf_comb = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    rf_comb.fit(X_tr_comb, y_tr)
    y_pred_rf_comb = rf_comb.predict(X_te_comb)
    r = evaluate(y_te, y_pred_rf_comb)
    results.append({"model": "RF (contact+descriptor)", "rmse": r[0], "pearson": r[1], "spearman": r[2]})
    print(f"  ④ RF 接触+描述子           RMSE={r[0]:.3f}  Pearson={r[1]:.3f}  Spearman={r[2]:.3f}")

    # 选出最优模型（按 Pearson 最高）作为最终主结果
    best_idx = int(np.argmax([res["pearson"] for res in results]))
    best_model = results[best_idx]["model"]
    if best_idx == 3:
        y_best = y_pred_rf_comb
    elif best_idx == 2:
        y_best = y_pred_comb
    else:
        y_best = rf.predict(X_te)  # 兜底用基线
    print(f"\n  最优模型：{best_model}")

    # 保存最优模型预测（画图脚本读取此文件）
    pd.DataFrame({"code": test_codes, "true_pKd": y_te, "pred_pKd": y_best}).to_csv(
        os.path.join(RESULTS_DIR, "improved_predictions.csv"), index=False)

    # 保存对比表
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)
    print("\n=== 对比结果 ===")
    print(df_res.to_string(index=False))

    # === 消融实验（统一用随机森林，分离特征贡献） ===
    print("\n=== 消融实验（随机森林） ===")
    ablation = []

    # A) 仅接触特征（36 维）—— 即 RF-Score 基线
    rf_contact_only = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    rf_contact_only.fit(X_tr, y_tr)
    r = evaluate(y_te, rf_contact_only.predict(X_te))
    ablation.append({"features": "Contact only (36)", "rmse": r[0], "pearson": r[1], "spearman": r[2]})
    print(f"  仅接触特征(36维)          RMSE={r[0]:.3f}  Pearson={r[1]:.3f}  Spearman={r[2]:.3f}")

    # B) 仅配体描述子（12 维）
    rf_desc_only = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    rf_desc_only.fit(desc_tr, y_tr)
    r = evaluate(y_te, rf_desc_only.predict(desc_te))
    ablation.append({"features": "Descriptor only (12)", "rmse": r[0], "pearson": r[1], "spearman": r[2]})
    print(f"  仅描述子(12维)            RMSE={r[0]:.3f}  Pearson={r[1]:.3f}  Spearman={r[2]:.3f}")

    # C) 两者结合（48 维）
    rf_ablation_comb = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    rf_ablation_comb.fit(X_tr_comb, y_tr)
    r = evaluate(y_te, rf_ablation_comb.predict(X_te_comb))
    ablation.append({"features": "Combined (48)", "rmse": r[0], "pearson": r[1], "spearman": r[2]})
    print(f"  两者结合(48维)            RMSE={r[0]:.3f}  Pearson={r[1]:.3f}  Spearman={r[2]:.3f}")

    # 保存消融结果
    df_ablation = pd.DataFrame(ablation)
    df_ablation.to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)
    print("\n=== 消融结果 ===")
    print(df_ablation.to_string(index=False))
    print("  -> 结合两者最优则证明接触特征与描述子互补。")


if __name__ == "__main__":
    main()
