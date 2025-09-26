import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("./dataset/PSM/train.csv")

# 写入 Parquet（默认使用 Snappy 压缩）
df.to_parquet("output.parquet", engine='pyarrow')

# 可选：指定压缩算法（如 gzip）
df.to_parquet("output_gzip.parquet", engine='pyarrow', compression='gzip')