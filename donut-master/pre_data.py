import numpy as np
from donut import complete_timestamp, standardize_kpi

data = np.loadtxt('sample_data/g.csv', delimiter=',', skiprows=1, unpack=True)
timestamp, values, labels = data[0], data[1], data[2].astype(np.int32)
# 读取原始数据

# 如果没有标签，使用全零数组
#labels = np.zeros_like(values, dtype=np.int32)

# 补全时间戳，获取缺失点指示器
timestamp, missing, (values, labels) = \
    complete_timestamp(timestamp, (values, labels))

# 分割训练和测试数据
test_portion = 0.3
test_n = int(len(values) * test_portion)
train_values, test_values = values[:-test_n], values[-test_n:]
train_labels, test_labels = labels[:-test_n], labels[-test_n:]
train_missing, test_missing = missing[:-test_n], missing[-test_n:]

# 标准化训练和测试数据
train_values, mean, std = standardize_kpi(
    train_values, excludes=np.logical_or(train_labels, train_missing))
test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)