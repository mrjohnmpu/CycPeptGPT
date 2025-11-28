from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import os

def is_cyclic_peptide(mol, min_ring_size=12):
    """
    判断一个分子是否为环肽（大环）。
    标准：
    1. 必须包含一个大小 >= min_ring_size 的环 (通常定义为12或更大)。
    2. (可选) 检查是否有酰胺键结构，这里主要关注是否为大环。
    """
    if mol is None:
        return False
        
    try:
        # 获取最小环集 (SSSR)
        sssr = Chem.GetSymmSSSR(mol)
        
        # 检查是否有任何一个环的尺寸大于等于阈值
        for ring in sssr:
            if len(ring) >= min_ring_size:
                return True
        return False
    except:
        return False

def analyze_and_filter_cyclic(file_path, output_path=None):
    print(f"--- Processing: {file_path} ---")
    
    # 1. 读取数据
    # 自动处理可能存在的 header 或 sep 问题
    try:
        # 尝试作为空格分隔读取，假设没有 header
        df = pd.read_csv(file_path, header=None, names=['SMILES'], sep=' ') 
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    total = len(df)
    valid_mols = []
    
    # 2. 合法性检查与标准化
    # 使用 Set 来去重，提高效率
    unique_canonical_smiles = set()
    cyclic_smiles = []

    print("Validating and Filtering...")
    
    valid_count = 0
    for smi in df['SMILES']:
        # 容错处理：确保输入是字符串
        if not isinstance(smi, str):
            continue
            
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_count += 1
            # 标准化 SMILES
            can_smi = Chem.MolToSmiles(mol, canonical=True)
            
            # 3. 核心修改：筛选环肽
            # 只有当它是合法分子，且是大环，且之前没出现过（去重）时，才加入列表
            if is_cyclic_peptide(mol, min_ring_size=12):
                if can_smi not in unique_canonical_smiles:
                    unique_canonical_smiles.add(can_smi)
                    cyclic_smiles.append(can_smi)

    # 统计数据
    cyclic_count = len(cyclic_smiles)
    
    print(f"Total input SMILES: {total}")
    print(f"Valid SMILES: {valid_count} ({valid_count/total:.2%})")
    print(f"Unique Cyclic Peptides (Macrocycles >= 12): {cyclic_count}")
    
    if valid_count > 0:
        print(f"Cyclic Ratio (among valid): {cyclic_count / valid_count:.2%}")

    # 4. 保存结果
    if output_path is None:
        # 自动生成文件名：原文件名_cyclic_only.csv
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_cyclic_only{ext}"
    
    # 保存只包含唯一环肽的数据框
    out_df = pd.DataFrame(cyclic_smiles, columns=['SMILES'])
    out_df.to_csv(output_path, index=False)
    print(f"Saved filtered cyclic peptides to: {output_path}")

# --- 使用示例 ---
input_file = '../output/generate/prompt_generate/cyc_prompt_generated_10_0.9_1_10000_1.0.csv'
# output_file = 'output/generate/prompt_generate/cyc_filtered_final.csv' # 你可以指定输出路径

analyze_and_filter_cyclic(input_file)


# from rdkit import Chem
# from rdkit.Chem import Descriptors
# import pandas as pd

# def analyze_generated_smiles(file_path):
#     # 1. 读取数据
#     # 假设文件没header，第一列是SMILES
#     df = pd.read_csv(file_path, header=None, names=['SMILES'], sep=' ') 
    
#     valid_mols = []
#     valid_smiles = []
    
#     # 2. 合法性检查
#     for smi in df['SMILES']:
#         mol = Chem.MolFromSmiles(smi)
#         if mol is not None:
#             valid_mols.append(mol)
#             # 转为标准SMILES以便去重
#             valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            
#     total = len(df)
#     valid_count = len(valid_smiles)
#     unique_smiles = list(set(valid_smiles))
#     unique_count = len(unique_smiles)
    
#     print(f"Total generated: {total}")
#     print(f"Validity: {valid_count / total:.2%} ({valid_count}/{total})")
#     print(f"Uniqueness: {unique_count / valid_count:.2%} ({unique_count}/{valid_count})")
    
#     # 3. 简单环化检查 (Head-to-Tail 通常意味着骨架上的首尾原子成环)
#     # 这是一个复杂的判断，这里用简单逻辑演示：检查是否包含大环
#     cyclic_count = 0
#     for mol in valid_mols:
#         sssr = Chem.GetSymmSSSR(mol) # 获取最小环集
#         # 环肽通常有大环（例如 > 12个原子）
#         has_macrocycle = any(len(ring) >= 12 for ring in sssr)
#         if has_macrocycle:
#             cyclic_count += 1
            
#     print(f"Macrocycle Ratio (Approx): {cyclic_count / valid_count:.2%}")

#     # 4. 保存清洗后的数据
#     pd.DataFrame(unique_smiles, columns=['SMILES']).to_csv('cyc_prompt_generated_10_0.9_1_10000_1.0_Validity.csv', index=False)

# # 使用
# analyze_generated_smiles('output/generate/prompt_generate/cyc_prompt_generated_10_0.9_1_10000_1.0.csv')