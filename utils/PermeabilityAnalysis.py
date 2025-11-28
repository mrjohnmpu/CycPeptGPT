import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm  # 进度条库，需安装：pip install tqdm

# ==========================================
# 1. 核心预测函数 (基于你提供的代码)
# ==========================================
def predict_permeability_single(peptide_smiles, model, scalar):
    """
    预测单个SMILES的渗透性。
    为了效率，将模型作为参数传入，避免每次调用都重新加载模型。
    """
    try:
        mol = Chem.MolFromSmiles(peptide_smiles)
        if mol is None:
            return None
            
        # Calculate Morgan fingerprint
        fps = AllChem.GetMorganFingerprint(mol, radius=4, useChirality=True, useCounts=True)
        
        # Convert to array format (2048 bits)
        size = 2048 
        arr = np.zeros((size,), np.int32)
        for idx, v in fps.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        
        # Transform using scalar
        x_test = scalar.transform([arr])
        
        # Make prediction
        predictions = model.predict_proba(x_test)
        return predictions[0, 1]  # Return probability of positive class (or value depending on model)
        
    except Exception as e:
        # 生产环境中可以注释掉打印，避免刷屏
        # print(f"Error predicting for {peptide_smiles[:10]}...: {str(e)}")
        return None

def load_models():
    """加载模型和Scalar"""
    # 请根据你的实际目录结构修改这里的路径
    # base_path = os.getcwd() 
    base_path = "../"
    model_path = os.path.join(base_path, "model", "pepinvent", "models", "predictive_model.pckl")
    scalar_path = os.path.join(base_path, "model", "pepinvent", "models", "feature_scalar.pckl")
    # model_path = os.path.join("model", "pepinvent", "models", "predictive_model.pckl")
    # scalar_path = os.path.join("model", "pepinvent", "models", "feature_scalar.pckl")
    if not os.path.exists(model_path) or not os.path.exists(scalar_path):
        raise FileNotFoundError(f"Model files not found at:\n{model_path}\n{scalar_path}")

    print("Loading models...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scalar_path, 'rb') as f:
        scalar = pickle.load(f)
    print("Models loaded successfully.")
    return model, scalar

# ==========================================
# 2. 批量处理流程
# ==========================================
def batch_predict_and_save(input_csv, output_csv):
    """
    读取输入CSV,预测渗透性,保存结果。
    如果 output_csv 已存在，则直接读取，跳过预测。
    """
    if os.path.exists(output_csv):
        print(f"Found existing results at: {output_csv}")
        print("Loading directly to save time...")
        df = pd.read_csv(output_csv)
        return df

    print(f"Reading input data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 确保有 SMILES 列
    if 'SMILES' not in df.columns:
        # 如果没有 header，尝试假设第一列是 SMILES
        if len(df.columns) >= 1:
            df.rename(columns={df.columns[0]: 'SMILES'}, inplace=True)
        else:
            raise ValueError("CSV format error: Cannot find SMILES column.")

    # 加载模型
    model, scalar = load_models()
    
    print("Starting prediction...")
    # 使用 list comprehension 和 tqdm 显示进度
    scores = []
    smiles_list = df['SMILES'].tolist()
    
    for smi in tqdm(smiles_list, desc="Predicting"):
        score = predict_permeability_single(smi, model, scalar)
        scores.append(score)
    
    df['Permeability'] = scores
    
    # 移除预测失败（None）的行
    df_clean = df.dropna(subset=['Permeability'])
    
    print(f"Saving results to: {output_csv}")
    df_clean.to_csv(output_csv, index=False)
    
    return df_clean

# ==========================================
# 3. 绘图函数 (复刻你的参考图风格)
# ==========================================
def plot_permeability_distribution(df, save_plot_path='permeability_plot.png'):
    # 设置暗色背景风格
    # plt.style.use('dark_background')
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 获取数据
    data = df['Permeability']
    
    # 绘制 KDE (核密度估计) 图
    # color参数可以调整为你喜欢的颜色，参考图中使用了类似蓝绿色
    # fill=True 填充颜色
    # alpha 设置透明度
    sns.kdeplot(data, 
                color="#69b3a2", 
                fill=True, 
                alpha=0.4, 
                linewidth=2, 
                ax=ax, 
                label='Generated Cyclic Peptides')

    # 如果你有其他基准数据（比如 Baseline 或 Chembl），可以在这里加载并叠加绘制
    # sns.kdeplot(baseline_data, color="#407294", fill=True, label='Baseline')

    # 添加垂直虚线 (X = -6.0)
    # cutoff_value = -6.0
    # ax.axvline(x=cutoff_value, color='#d66060', linestyle='--', linewidth=2, alpha=0.8)

    # 添加箭头和文字注释 "X = -6.0 ->"
    # xy是箭头尖端的位置，xytext是文字的位置
    # ax.annotate(f'X = {cutoff_value}', 
    #             xy=(cutoff_value, 0.15),  # 箭头指向 (x, y)
    #             xytext=(cutoff_value - 1.5, 0.15), # 文字位置
    #             color='#cccccc',
    #             fontsize=14,
    #             arrowprops=dict(facecolor='#cccccc', shrink=0.05, width=2, headwidth=8))

    # 设置坐标轴标签和标题
    ax.set_xlabel('Permeability', fontsize=16, color='#cccccc')
    ax.set_ylabel('Density', fontsize=16, color='#cccccc')
    
    # 调整坐标轴范围 (根据你的参考图调整，例如 -10 到 -4)
    # 如果数据超出这个范围，请适当调整或注释掉
    # ax.set_xlim(-10, -4)
    
    # 去除上方和右侧的边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['left'].set_color('#cccccc')
    
    # 调整刻度字体大小
    ax.tick_params(axis='both', colors='#cccccc', labelsize=14)

    # 添加图例 (去除边框)
    legend = ax.legend(frameon=False, fontsize=12)
    plt.setp(legend.get_texts(), color='#cccccc')

    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_plot_path}")
    plt.show()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 输入文件路径 (上一步生成的只包含环肽的文件)
    input_file = '../output/generate/prompt_generate/cyc_prompt_generated_10_0.9_1_10000_1.0_cyclic_only.csv'
    
    # 2. 结果保存路径 (保存预测值，方便下次直接读取)
    results_file = '../output/generate/prompt_generate/cyc_prompt_generated_10_0.9_1_10000_1.0_cyclic_only_with_permeability.csv'
    
    # 3. 执行流程
    # 步骤 A: 预测或加载
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found. Please run the filtering script first.")
    else:
        df_results = batch_predict_and_save(input_file, results_file)
        
        # 步骤 B: 绘图
        if not df_results.empty:
            plot_permeability_distribution(df_results)
        else:
            print("No valid data to plot.")