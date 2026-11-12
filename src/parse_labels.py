"""
parse_labels.py
解析 PDBbind refined set 和 CASF-2016 的索引文件，提取 (PDB code, pKd) 标签对。

输出：
  results/refined_labels.csv   -> columns: code, pKd
  results/casf_labels.csv      -> columns: code, pKd
"""
import os
import re
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data set")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_refined_index(path):
    """解析 INDEX_refined_data.2020，返回 [(code, pKd), ...]"""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            code = parts[0]
            try:
                pkd = float(parts[3])  # 第4列是 -logKd/Ki
            except ValueError:
                continue
            rows.append((code, pkd))
    return rows


def parse_casf_index(path):
    """解析 CASF-2016 CoreSet.dat，返回 [(code, pKd), ...]"""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            code = parts[0]
            try:
                pkd = float(parts[3])  # logKa 列
            except ValueError:
                continue
            rows.append((code, pkd))
    return rows


def main():
    refined_idx = os.path.join(DATA_DIR, "refined-set", "index", "INDEX_refined_data.2020")
    casf_idx = os.path.join(DATA_DIR, "CASF-2016", "power_scoring", "CoreSet.dat")

    refined_rows = parse_refined_index(refined_idx)
    casf_rows = parse_casf_index(casf_idx)

    df_refined = pd.DataFrame(refined_rows, columns=["code", "pKd"])
    df_casf = pd.DataFrame(casf_rows, columns=["code", "pKd"])

    refined_out = os.path.join(RESULTS_DIR, "refined_labels.csv")
    casf_out = os.path.join(RESULTS_DIR, "casf_labels.csv")
    df_refined.to_csv(refined_out, index=False)
    df_casf.to_csv(casf_out, index=False)

    print(f"[refined] 解析 {len(df_refined)} 个复合物 -> {refined_out}")
    print(f"[CASF]    解析 {len(df_casf)} 个复合物 -> {casf_out}")
    print("\nrefined pKd 分布：")
    print(df_refined["pKd"].describe())
    print("\nCASF pKd 分布：")
    print(df_casf["pKd"].describe())


if __name__ == "__main__":
    main()
