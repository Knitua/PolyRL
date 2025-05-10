import pandas as pd

def excel_to_txt(excel_file, txt_file, column_name='Smiles'):
    # 读取 Excel 文件
    df = pd.read_csv(excel_file)
    
    # 检查是否存在指定的列
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the Excel file.")
    
    # 获取指定列的 SMILES 数据
    smiles_list = df[column_name].dropna().tolist()  # 去除空值
    
    # 将 SMILES 数据写入 .txt 文件
    with open(txt_file, 'w') as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")

# 示例用法
excel_file = "datasetB.csv"  # 你的 Excel 文件路径
txt_file = "datasetB.txt"         # 转换后的 .txt 文件路径
excel_to_txt(excel_file, txt_file)
