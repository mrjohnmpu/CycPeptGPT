# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem import AllChem

# def draw_cyclic_peptides_grid(csv_path, output_image="cyclic_peptides_grid.png", max_mols=20):
#     # 1. 读取数据
#     try:
#         df = pd.read_csv(csv_path)
#         # 确保列名正确，如果不是 'SMILES' 请修改
#         smiles_list = df['SMILES'].tolist()
#     except Exception as e:
#         print(f"读取CSV失败: {e}")
#         return

#     mols = []
#     legends = []
    
#     print(f"正在处理前 {max_mols} 个分子...")
    
#     # 2. 转换为 RDKit 分子对象
#     for i, smi in enumerate(smiles_list[:max_mols]):
#         mol = Chem.MolFromSmiles(smi)
#         if mol:
#             # 生成 2D 坐标，这对于环肽展示非常重要
#             AllChem.Compute2DCoords(mol)
#             mols.append(mol)
#             legends.append(f"Seq_{i}")

#     # 3. 绘制网格图
#     if mols:
#         img = Draw.MolsToGridImage(
#             mols, 
#             molsPerRow=4,              # 每行显示几个
#             subImgSize=(400, 400),     # 每个小图的尺寸 (环肽比较大，建议设置大一点)
#             legends=legends,           # 显示每个分子的标签
#             useSVG=False               # 设置为 True 可以保存为矢量图
#         )
        
#         # 保存图片
#         img.save(output_image)
#         print(f"成功保存网格图至: {output_image}")
#     else:
#         print("没有有效的分子可以绘制。")

# # 使用示例
# # 请将 'clean_generated.csv' 替换为您清洗后保存的文件名
# draw_cyclic_peptides_grid('utils/clean_generated.csv', max_mols=16)


# import os
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem import AllChem

# def highlight_and_save_individual(csv_path, output_dir="output/generate/prompt_generate/peptide_images", max_mols=10):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     df = pd.read_csv(csv_path)
    
#     print(f"开始绘制并高亮环结构，输出目录: {output_dir}")

#     for i, smi in enumerate(df['SMILES'][:max_mols]):
#         mol = Chem.MolFromSmiles(smi)
#         if not mol:
#             continue
            
#         # 1. 计算 2D 坐标
#         AllChem.Compute2DCoords(mol)
        
#         # 2. 寻找最大环 (Macrocycle) 进行高亮
#         # 获取最小环集 (SSSR)
#         sssr = Chem.GetSymmSSSR(mol)
        
#         # 找到原子数大于等于 10 的大环 (环肽通常很大)
#         macrocycle_atoms = set()
#         for ring in sssr:
#             if len(ring) >= 10: 
#                 macrocycle_atoms.update(ring)
        
#         # 将集合转为列表
#         highlight_atoms = list(macrocycle_atoms)
        
#         # 3. 绘图配置
#         d = Draw.rdMolDraw2D.MolDraw2DSVG(500, 500) # 使用 Cairo 引擎绘制高质量 PNG
        
#         # 设置绘图选项
#         opts = d.drawOptions()
#         opts.addAtomIndices = False # 不显示原子索引，保持图片干净
        
#         # 绘制
#         d.DrawMolecule(mol, highlightAtoms=highlight_atoms)
#         d.FinishDrawing()

#         svg_text = d.GetDrawingText()
        
#         # 保存
#         output_file = os.path.join(output_dir, f"peptide_{i}_highlighted.svg")
#         # d.WriteDrawingText(output_file)
#         with open(output_file, 'w') as f:
#             f.write(svg_text)
        
#     print("绘制完成！")

# # 使用示例
# highlight_and_save_individual('utils/clean_generated.csv', max_mols=1)

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def draw_highlighted_grid_svg(csv_path, output_dir, output_file="ocyclic_peptides_grid_highlighted.svg", mols_per_row=4, max_mols=20):
    """
    在一张 SVG 网格图中绘制多个分子，并自动高亮检测到的大环结构。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 1. 读取数据
    try:
        df = pd.read_csv(csv_path)
        smiles_list = df['SMILES'].tolist()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    mols = []
    highlight_lists = []  # 这是一个列表的列表，存储每个分子需要高亮的原子索引
    legends = []

    print(f"正在处理前 {min(len(smiles_list), max_mols)} 个分子...")

    # 2. 处理每个分子
    for i, smi in enumerate(smiles_list[:max_mols]):
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue

        # A. 生成 2D 坐标 (对于环肽非常重要，防止结构重叠)
        AllChem.Compute2DCoords(mol)

        # B. 寻找大环 (Macrocycle)
        # 获取最小环集 (SSSR)
        sssr = Chem.GetSymmSSSR(mol)
        
        current_mol_highlights = set()
        for ring in sssr:
            # 阈值设为 10 或 12，视您的肽大小而定
            if len(ring) >= 10: 
                current_mol_highlights.update(ring)
        
        # 将集合转为列表加入总的高亮列表
        # 注意：即使没有找到大环，也要加入一个空列表 []，以保持索引对齐
        highlight_lists.append(list(current_mol_highlights))
        
        # C. 添加到分子列表
        mols.append(mol)
        legends.append(f"Peptide_{i}")

    if not mols:
        print("没有有效的分子可绘制。")
        return

    # 3. 配置绘图选项
    # 这一步是为了让生成的 SVG 更好看（可选）
    draw_options = Draw.rdMolDraw2D.MolDrawOptions()
    draw_options.addAtomIndices = False
    draw_options.setBackgroundColour((1, 1, 1))
    # 可以在这里设置高亮颜色，默认是浅红色

    # 4. 生成网格 SVG 字符串
    # MolsToGridImage 当 useSVG=True 时，返回的是 SVG 文本字符串
    svg_grid = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(450, 450),        # 环肽较大，建议子图尺寸设大一点
        legends=legends,
        highlightAtomLists=highlight_lists,  # 【关键】传入高亮列表
        useSVG=True,                  # 【关键】输出 SVG 格式
        drawOptions=draw_options      # 应用绘图选项
    )

    # 5. 保存到文件
    output_file = os.path.join(output_dir, output_file)
    try:
        with open(output_file, "w") as f:
            f.write(svg_grid)
        print(f"✅ 成功保存高亮网格图至: {output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")

# --- 运行 ---
# 请确保路径正确
draw_highlighted_grid_svg('utils/clean_generated.csv', output_dir='output/generate/prompt_generate/peptide_images', max_mols=16)