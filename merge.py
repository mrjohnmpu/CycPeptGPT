import pandas as pd
import os

def merge_smiles_to_txt(csv_files, output_txt_file):
    """
    從多個 CSV 檔案中讀取 'SMILES' 列，並將所有 SMILES 序列合併到一個 TXT 檔案中。

    Args:
        csv_files (list): 包含輸入 CSV 檔案路徑的列表。
        output_txt_file (str): 輸出 TXT 檔案的路徑。
    """
    all_smiles = []
    total_count = 0

    print("開始合併 SMILES 序列...")

    for file_path in csv_files:
        if not os.path.exists(file_path):
            print(f"警告：找不到檔案 '{file_path}'，已跳過。")
            continue

        try:
            print(f"正在讀取檔案：'{file_path}'...")
            df = pd.read_csv(file_path)

            if 'SMILES' in df.columns:
                # 將此檔案中的所有 SMILES 添加到主列表中
                smiles_list = df['SMILES'].dropna().astype(str).tolist()
                all_smiles.extend(smiles_list)
                print(f" -> 從 '{os.path.basename(file_path)}' 成功提取 {len(smiles_list)} 個 SMILES 序列。")
            else:
                print(f"警告：在檔案 '{file_path}' 中找不到 'SMILES' 列，已跳過。")

        except Exception as e:
            print(f"處理檔案 '{file_path}' 時發生錯誤：{e}")

    # 將所有 SMILES 序列寫入到 TXT 檔案中，每個佔一行
    try:
        print(f"\n正在將所有序列寫入到 '{output_txt_file}'...")
        with open(output_txt_file, 'w', encoding='utf-8') as f:
            for smiles in all_smiles:
                f.write(smiles + '\n')
        
        total_count = len(all_smiles)
        print(f"處理完成！")
        print(f"總共 {total_count} 個 SMILES 序列已成功寫入到 '{output_txt_file}'。")

    except Exception as e:
        print(f"寫入到檔案 '{output_txt_file}' 時發生錯誤：{e}")


if __name__ == '__main__':
    # --- 設定您的檔案路徑 ---

    # 假設您的 CSV 檔案都在 'data/training_data/' 這個資料夾下
    input_directory = 'data/'
    
    # 要讀取的 CSV 檔案列表
    input_csv_files = [
        os.path.join(input_directory, 'restored_train.csv'),
        os.path.join(input_directory, 'restored_validation.csv'),
        os.path.join(input_directory, 'restored_test.csv')
    ]

    # 定義輸出的 TXT 檔案名稱
    output_txt_path = os.path.join(input_directory, 'all_smiles_for_tokenizer.txt')

    # --- 執行合併功能 ---
    merge_smiles_to_txt(input_csv_files, output_txt_path)