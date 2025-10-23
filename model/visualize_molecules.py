import pandas as pd
import os
import math
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger


def check_if_cyclic(mol):
    """
    检查一个 RDKit 分子对象是否包含环。
    """
    if mol is None:
        return False
    # GetSSSR 返回“最小环集合”中的环的数量
    # 如果数量大于0，说明分子中存在环
    return Chem.GetSSSR(mol) > 0


def visualize_generated_smiles(csv_path, output_dir):
    """
    读取生成的SMILES CSV文件，验证它们，并可视化所有有效的环状肽。
    """

    print("--- 开始可视化和分析 ---")

    # 禁用 RDKit 的冗余错误日志 (SMILES 解析失败时会打印很多)
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    os.makedirs(output_dir, exist_ok=True)

    # 读取 CSV 文件
    try:
        df = pd.read_csv(csv_path, header=None, sep=' ')
        # 假设SMILES在第一列 (索引 0)
        smiles_list = df[0].astype(str).tolist()
        print(f"成功读取 {len(smiles_list)} 个SMILES序列。")
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{csv_path}'")
        print("请确保你已经运行了生成脚本，并且路径正确。")
        return
    except Exception as e:
        print(f"读取CSV时出错: {e}")
        return

    # 4. 分析和过滤 SMILES
    valid_mols = []
    legends = []
    invalid_smiles_count = 0
    non_cyclic_count = 0

    # 使用 set 来跟踪唯一的 SMILES，防止重复
    unique_smiles = set()

    for i, smiles in enumerate(smiles_list):
        if smiles in unique_smiles:
            continue
        unique_smiles.add(smiles)

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # SMILES 语法无效
            invalid_smiles_count += 1
        else:
            if check_if_cyclic(mol):
                # 这是一个有效的、环状的分子
                valid_mols.append(mol)
                legends.append(f"ID {i}\n{smiles[:40]}...")  # 图例显示部分SMILES
            else:
                # 这是一个有效的、但非环状（线性）的分子
                non_cyclic_count += 1

    # 5. 打印分析报告
    print("\n--- 分析报告 ---")
    print(f"总共SMILES:    {len(smiles_list)}")
    print(f"唯一SMILES:    {len(unique_smiles)}")
    print(f"无效SMILES:    {invalid_smiles_count} (语法错误)")
    print(f"有效(线性):    {non_cyclic_count}")
    print(f"有效(环状):    {len(valid_mols)}")
    print("--------------------")

    if not valid_mols:
        print("没有找到有效的环状分子进行可视化。")
        return

    # 6. 将所有有效的环状分子绘制到网格图中并保存
    # (为了防止图像过大，我们进行分页)

    mols_per_page = 100  # 每张图显示100个分子
    mols_per_row = 10
    img_size = (250, 250)

    num_pages = math.ceil(len(valid_mols) / mols_per_page)
    print(f"\n正在生成 {num_pages} 页可视化图片，保存至 '{output_dir}'...")

    for page in range(num_pages):
        start_idx = page * mols_per_page
        end_idx = start_idx + mols_per_page

        mol_chunk = valid_mols[start_idx:end_idx]
        legend_chunk = legends[start_idx:end_idx]

        try:
            img = Draw.MolsToGridImage(
                mol_chunk,
                molsPerRow=mols_per_row,
                subImgSize=img_size,
                legends=legend_chunk
            )

            output_path = os.path.join(output_dir, f"cyclic_molecules_page_{page + 1}.png")
            img.save(output_path)
            print(f"已保存: {output_path}")

        except Exception as e:
            print(f"错误：在生成第 {page + 1} 页图像时失败: {e}")

    print("--- 可视化完成 ---")


if __name__ == "__main__":
    # 确保这个路径指向你 *生成* 的CSV文件
    INPUT_FILE = '../output/generate_cyc_seq.csv'

    # 可视化图片将保存到这里
    OUTPUT_IMAGE_DIR = '../output/visualizations'

    visualize_generated_smiles(INPUT_FILE, OUTPUT_IMAGE_DIR)