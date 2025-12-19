import pandas as pd

# 定义文件路径
file_path = '../data/index/numericPPG_index.csv'

try:
    # 读取 CSV 文件
    # 如果文件包含中文，可能需要指定 encoding='utf-8' 或 'gbk'
    df = pd.read_csv(file_path)
    
    # 显示前 5 行数据
    print("文件的前5行内容如下：")
    print(df.head())
    
    # 如果需要查看更多行，例如前 10 行，可以使用 df.head(10)
    
    # 打印数据的基本信息（列名、数据类型等）
    print("\n数据基本信息：")
    print(df.info())

except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}，请检查路径是否正确。")
except Exception as e:
    print(f"读取文件时发生错误：{e}")
