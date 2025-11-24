import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# 设定输入文件和输出目录
INPUT_CSV = '../output/generate/prompt_generate/cyc_prompt_topk_500_64.csv'
OUTPUT_DIR = '../output/generate/prompt_generate/peptide_images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def is_cyclic_peptide(mol, min_ring_size=9):
    """
    判断是否为环肽。
    逻辑：检查是否存在大小 >= min_ring_size 的环。
    常见的侧链环（Phe, Tyr, Trp, His, Pro）大小为 5 或 6。
    """
    if mol is None:
        return False, "Invalid Molecule"

    # 获取所有环的信息
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if not atom_rings:
        return False, "Linear (No Rings)"

    # 检查是否有大环
    max_ring_size = 0
    for ring in atom_rings:
        ring_size = len(ring)
        if ring_size > max_ring_size:
            max_ring_size = ring_size

    if max_ring_size >= min_ring_size:
        return True, f"Cyclic (Max Ring Size: {max_ring_size})"
    else:
        return False, f"Linear (Max Ring Size: {max_ring_size} - Likely Sidechain)"


def process_peptides(csv_path):
    # 读取 CSV (假设没有表头，第一列是 SMILES)
    # 如果你的 CSV 有表头，请去掉 header=None，并指定列名
    df = pd.read_csv(csv_path, header=None, sep=' ')
    # 假设第一列是 SMILES。如果有多个列，请修改这里的索引 [0]
    smiles_list = df[0].tolist()

    print(f"Total sequences to process: {len(smiles_list)}")

    valid_count = 0
    cyclic_count = 0

    for idx, smiles in enumerate(smiles_list):
        # 1. 转换为 RDKit 分子对象
        mol = Chem.MolFromSmiles(smiles)

        # 2. 检查有效性 (Generative Model 可能会生成无效的 SMILES)
        if mol is None:
            print(f"Sequence {idx}: Invalid SMILES")
            continue

        valid_count += 1

        # 3. 判断是否为环肽
        is_cyclic, status = is_cyclic_peptide(mol)

        if is_cyclic:
            cyclic_count += 1
            tag = "CYCLIC"
        else:
            tag = "LINEAR"

        # 4. 可视化并保存
        # 只有当前 50 个或者特定条件的才保存图片，避免生成几万张图
        if idx < 50 or is_cyclic:
            try:
                # 生成 2D 坐标让图片更好看
                AllChem.Compute2DCoords(mol)

                # 图片文件名: id_类型_环大小.png
                img_name = f"{idx}_{tag}_{status.replace(' ', '_')}.png"
                img_path = os.path.join(OUTPUT_DIR, img_name)

                # 绘图
                Draw.MolToFile(mol, img_path, size=(400, 400), legend=status)
            except Exception as e:
                print(f"Error drawing {idx}: {e}")

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Valid SMILES: {valid_count} / {len(smiles_list)}")
    print(f"Predicted Cyclic Peptides: {cyclic_count}")
    print(f"Images saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    process_peptides(INPUT_CSV)