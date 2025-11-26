from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

def analyze_generated_smiles(file_path):
    # 1. 读取数据
    # 假设文件没header，第一列是SMILES
    df = pd.read_csv(file_path, header=None, names=['SMILES'], sep=' ') 
    
    valid_mols = []
    valid_smiles = []
    
    # 2. 合法性检查
    for smi in df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)
            # 转为标准SMILES以便去重
            valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            
    total = len(df)
    valid_count = len(valid_smiles)
    unique_smiles = list(set(valid_smiles))
    unique_count = len(unique_smiles)
    
    print(f"Total generated: {total}")
    print(f"Validity: {valid_count / total:.2%} ({valid_count}/{total})")
    print(f"Uniqueness: {unique_count / valid_count:.2%} ({unique_count}/{valid_count})")
    
    # 3. 简单环化检查 (Head-to-Tail 通常意味着骨架上的首尾原子成环)
    # 这是一个复杂的判断，这里用简单逻辑演示：检查是否包含大环
    cyclic_count = 0
    for mol in valid_mols:
        sssr = Chem.GetSymmSSSR(mol) # 获取最小环集
        # 环肽通常有大环（例如 > 12个原子）
        has_macrocycle = any(len(ring) >= 12 for ring in sssr)
        if has_macrocycle:
            cyclic_count += 1
            
    print(f"Macrocycle Ratio (Approx): {cyclic_count / valid_count:.2%}")

    # 4. 保存清洗后的数据
    pd.DataFrame(unique_smiles, columns=['SMILES']).to_csv('clean_generated.csv', index=False)

# 使用
analyze_generated_smiles('output/generate/prompt_generate/cyc_prompt_generated_from_ckpt.csv')