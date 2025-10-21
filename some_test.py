import pandas as pd

def get_max_smiles_length(file_path, column_name='SMILES'):
    """
    讀取CSV文件並計算指定列中字符串的最大長度。

    Args:
        file_path (str): CSV文件的路徑。
        column_name (str): 包含SMILES字符串的列名。

    Returns:
        int: SMILES字符串的最大長度，如果文件或列不存在則返回0。
    """
    try:
        # 讀取CSV文件
        df = pd.read_csv(file_path)

        # 檢查指定的列是否存在
        if column_name not in df.columns:
            print(f"錯誤：在文件 '{file_path}' 中找不到名為 '{column_name}' 的列。")
            return 0

        # 計算SMILES列中每個字符串的長度，並找到最大值
        # .str.len() 會自動處理缺失值 (NaN)
        max_len = df[column_name].str.len().max()
        
        # 將可能出現的NaN轉換為0
        return int(max_len) if pd.notna(max_len) else 0

    except FileNotFoundError:
        print(f"錯誤：找不到文件 '{file_path}'。")
        return 0
    except Exception as e:
        print(f"處理文件時發生錯誤：{e}")
        return 0

if __name__ == '__main__':
    csv_file = './data/restored_train.csv'  # 將此處替換為您的實際文件名
    
    print(f"正在分析文件: '{csv_file}'...")
    
    max_length = get_max_smiles_length(csv_file)
    
    if max_length > 0:
        print(f"文件中SMILES序列的最大長度是: {max_length}")
