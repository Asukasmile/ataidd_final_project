"""
extract_features.py
RF-Score 特征提取：对每个蛋白-配体复合物，计算 36 维原子对接触特征。

RF-Score v1 原始定义（Ballester & Mitchell 2010）：
  将蛋白原子和配体原子按元素类型分组（各 6 类），
  对每一类「蛋白类型 × 配体类型」组合，统计 12 Å 距离内的接触原子对数量。

原子类型映射（RF-Score 常用）：
  C, N, O, S, Halogen(X/Cl/Br/I/F), Other(其余元素)

输出：
  results/refined_features.csv  (n x 37：36特征 + code)
  results/casf_features.csv
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data set")
RESULTS_DIR = os.path.join(BASE, "results")

CUTOFF = 12.0  # 接触距离阈值（Å）

# 原子类型映射：返回类型索引 0~5
def atom_type(atomic_num):
    if atomic_num == 6:
        return 0  # C
    elif atomic_num == 7:
        return 1  # N
    elif atomic_num == 8:
        return 2  # O
    elif atomic_num == 16:
        return 3  # S
    elif atomic_num in (9, 17, 35, 53):  # F, Cl, Br, I
        return 4  # Halogen
    else:
        return 5  # Other


def load_ligand_atoms(ligand_path):
    """加载配体原子，返回 (元素类型数组, 坐标数组)。支持 .mol2 和 .sdf"""
    ext = os.path.splitext(ligand_path)[1].lower()
    mol = None
    if ext == ".mol2":
        mol = Chem.MolFromMol2File(ligand_path, removeHs=True, sanitize=False)
    elif ext == ".sdf":
        suppl = Chem.SDMolSupplier(ligand_path, removeHs=True, sanitize=False)
        mol = suppl[0] if len(suppl) > 0 else None
    if mol is None:
        return None, None
    conf = mol.GetConformer()
    types = []
    coords = []
    for atom in mol.GetAtoms():
        types.append(atom_type(atom.GetAtomicNum()))
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    return np.array(types), np.array(coords)


# 合法元素符号集合（用于校验），大写首字母形式
_ELEMENT_SYMBOLS = set()
for _z in range(1, 119):
    _sym = Chem.GetPeriodicTable().GetElementSymbol(_z)
    _ELEMENT_SYMBOLS.add(_sym)  # 如 'C', 'N', 'Fe', 'Cl'


def _parse_element(line):
    """稳健地从 PDB ATOM 行解析元素符号。

    PDB 标准：元素在列 77-78（右对齐）。
    但很多文件不规范，需要回退到 atom name 推断。
    返回元素符号（如 'C', 'N', 'O'）或 None。
    """
    candidates = []

    # 1) 标准元素列（列 77-78，索引 76:78）
    if len(line) >= 78:
        raw = line[76:78].strip()
        if raw:
            letters = "".join(ch for ch in raw if ch.isalpha())
            if letters:
                candidates.append(letters.capitalize())

    # 2) atom name（列 13-16，索引 12:16）推断
    if len(line) >= 16:
        atom_name = line[12:16].strip()
        if atom_name:
            name = atom_name.lstrip("0123456789")
            if name and name[0].isalpha() and name[0].isupper():
                first = name[0]
                if len(name) > 1 and name[1].islower():
                    candidates.append(first + name[1])
                else:
                    candidates.append(first)

    # 3) 返回第一个是合法元素符号的候选
    for c in candidates:
        if c in _ELEMENT_SYMBOLS:
            return c
    return None


def load_protein_atoms(protein_path):
    """加载蛋白原子（只取 ATOM/HETATM 记录，忽略 H），返回 (类型数组, 坐标数组)"""
    types = []
    coords = []
    pt = Chem.GetPeriodicTable()
    with open(protein_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            elem = _parse_element(line)
            if elem is None or elem.upper() == "H":
                continue
            atomic_num = pt.GetAtomicNumber(elem)
            if atomic_num <= 0:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            types.append(atom_type(atomic_num))
            coords.append([x, y, z])
    return np.array(types), np.array(coords)


def compute_rfscore_features(protein_path, ligand_path):
    """计算 36 维 RF-Score 特征"""
    p_types, p_coords = load_protein_atoms(protein_path)
    l_types, l_coords = load_ligand_atoms(ligand_path)
    if p_types is None or l_types is None or len(p_types) == 0 or len(l_types) == 0:
        return None

    feat = np.zeros(36, dtype=np.float64)
    # 对每个配体原子，找 12Å 内的蛋白原子，统计接触对
    for i, lt in enumerate(l_types):
        # 距离平方
        d2 = np.sum((p_coords - l_coords[i]) ** 2, axis=1)
        within = d2 <= (CUTOFF ** 2)
        if not within.any():
            continue
        for pt in p_types[within]:
            # 蛋白类型 pt (0~5) 作为行，配体类型 lt (0~5) 作为列
            idx = pt * 6 + lt
            feat[idx] += 1
    return feat


FEATURE_NAMES = []
for pt in ["C", "N", "O", "S", "X", "Oth"]:
    for lt in ["C", "N", "O", "S", "X", "Oth"]:
        FEATURE_NAMES.append(f"{pt}-{lt}")


def extract_for_set(subdir, label_csv, out_csv):
    """对给定数据集的每个复合物提取特征"""
    df_labels = pd.read_csv(label_csv)
    features = []
    codes = []
    skipped = []
    for _, row in df_labels.iterrows():
        code = row["code"]
        complex_dir = os.path.join(DATA_DIR, subdir, code)
        protein_path = os.path.join(complex_dir, f"{code}_protein.pdb")
        ligand_mol2 = os.path.join(complex_dir, f"{code}_ligand.mol2")
        ligand_sdf = os.path.join(complex_dir, f"{code}_ligand.sdf")

        if not os.path.exists(protein_path):
            skipped.append((code, "no protein"))
            continue
        ligand_path = ligand_mol2 if os.path.exists(ligand_mol2) else (
            ligand_sdf if os.path.exists(ligand_sdf) else None
        )
        if ligand_path is None:
            skipped.append((code, "no ligand"))
            continue

        try:
            feat = compute_rfscore_features(protein_path, ligand_path)
        except Exception as e:
            skipped.append((code, f"error: {type(e).__name__}"))
            continue
        if feat is None or feat.sum() == 0:
            skipped.append((code, "empty atoms"))
            continue
        features.append(feat)
        codes.append(code)

    X = np.vstack(features)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df.insert(0, "code", codes)
    df.to_csv(out_csv, index=False)
    print(f"[{subdir}] 成功提取 {len(codes)} 个复合物的特征 -> {out_csv}")
    if skipped:
        print(f"  跳过 {len(skipped)} 个（示例：{skipped[:3]}）")


def main():
    extract_for_set(
        "refined-set",
        os.path.join(RESULTS_DIR, "refined_labels.csv"),
        os.path.join(RESULTS_DIR, "refined_features.csv"),
    )
    extract_for_set(
        "CASF-2016/coreset",
        os.path.join(RESULTS_DIR, "casf_labels.csv"),
        os.path.join(RESULTS_DIR, "casf_features.csv"),
    )


if __name__ == "__main__":
    main()
