"""
baseline_rfscore.py
RF-Score 基线：随机森林在 36 维接触特征上预测 pKd。

训练集：refined set（约 5316 个）
测试集：CASF-2016 core set（285 个）—— 与原始 RF-Score 论文的评测口径一致

评估指标：
  RMSE, Pearson R, Spearman rho

输出：
  results/baseline_predictions.csv   (code, true_pKd, pred_pKd)
  控制台打印指标
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")


def load_data():
    train_X = pd.read_csv(os.path.join(RESULTS_DIR, "refined_features.csv"))
    test_X = pd.read_csv(os.path.join(RESULTS_DIR, "casf_features.csv"))
    train_y = pd.read_csv(os.path.join(RESULTS_DIR, "refined_labels.csv"))
    test_y = pd.read_csv(os.path.join(RESULTS_DIR, "casf_labels.csv"))

    # 按 code 对齐
    train_X = train_X.set_index("code")
    test_X = test_X.set_index("code")
    train_y = train_y.set_index("code")
    test_y = test_y.set_index("code")

    # 取交集（特征提取可能跳过了个别复合物）
    train_common = train_X.index.intersection(train_y.index)
    test_common = test_X.index.intersection(test_y.index)

    X_tr = train_X.loc[train_common].values
    y_tr = train_y.loc[train_common]["pKd"].values
    X_te = test_X.loc[test_common].values
    y_te = test_y.loc[test_common]["pKd"].values

    return X_tr, y_tr, X_te, y_te, list(test_common)


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pear = pearsonr(y_true, y_pred)[0]
    spear = spearmanr(y_true, y_pred)[0]
    print(f"[{name}] RMSE={rmse:.3f}  Pearson={pear:.3f}  Spearman={spear:.3f}")
    return {"name": name, "rmse": rmse, "pearson": pear, "spearman": spear}


def main():
    X_tr, y_tr, X_te, y_te, test_codes = load_data()
    print(f"训练集: {X_tr.shape[0]} 个复合物, 特征 {X_tr.shape[1]} 维")
    print(f"测试集: {X_te.shape[0]} 个复合物")

    # RF-Score 默认参数（论文常用 n_estimators=500）
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    evaluate(y_te, y_pred, "RF-Score baseline")

    # 保存预测结果
    df_out = pd.DataFrame({
        "code": test_codes,
        "true_pKd": y_te,
        "pred_pKd": y_pred,
    })
    out_path = os.path.join(RESULTS_DIR, "baseline_predictions.csv")
    df_out.to_csv(out_path, index=False)
    print(f"预测结果已保存 -> {out_path}")


if __name__ == "__main__":
    main()
