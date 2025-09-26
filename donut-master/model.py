import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as K
from donut import complete_timestamp, standardize_kpi
from donut import Donut
from donut import DonutTrainer, DonutPredictor
from tfsnippet.modules import Sequential
from tfsnippet.utils import get_variables_as_dict, VariableSaver
from sklearn.metrics import precision_score, recall_score, f1_score
import glob
import pandas as pd
import time

tf.compat.v1.disable_v2_behavior()

# 设置保存目录
save_dir = "./model_save"
os.makedirs(save_dir, exist_ok=True)

# 创建用于存储结果的列表
results_summary = []

# 获取sample_data文件夹中的所有csv文件
csv_files = sorted(glob.glob('sample_data/*.csv'), key=os.path.getsize)

# 检查是否有找到CSV文件
if not csv_files:
    print("未找到任何CSV文件，请检查sample_data文件夹")
    exit()

print(f"找到 {len(csv_files)} 个CSV文件:")
for file in csv_files:
    print(f"  - {os.path.basename(file)}")


    
    
for csv_file_path in csv_files:
    for i in range(5):
        print(f"\n第 {i+1} 轮处理开始")
        round_results = []
        print(f"Processing file: {csv_file_path}")
        
        try:
            # 为每个文件创建新的TensorFlow会话
            tf.reset_default_graph()
            
            # 读取数据
            data = np.loadtxt(csv_file_path, delimiter=',', skiprows=1, unpack=True, usecols=[0, 1, 2])
            timestamp, values, labels = data[0], data[1], data[2].astype(np.int32)
            
            # 补全时间戳，获取缺失点指示器
            timestamp, missing, (values, labels) = complete_timestamp(timestamp, (values, labels))

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

            # 构建模型
            with tf.variable_scope('model') as model_vs:
                model = Donut(
                    h_for_p_x=Sequential([
                        K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                                    activation=tf.nn.relu),
                        K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                                    activation=tf.nn.relu),
                    ]),
                    h_for_q_z=Sequential([
                        K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                                    activation=tf.nn.relu),
                        K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                                    activation=tf.nn.relu),
                    ]),
                    x_dims=128,
                    z_dims=5,
                )

            # 创建训练器和预测器
            trainer = DonutTrainer(model=model, model_vs=model_vs, max_epoch=50)  # 减少训练轮数以加快速度
            predictor = DonutPredictor(model)

            # 在现有代码的相应位置添加以下内容
            with tf.Session() as sess:
                # 记录训练开始时间
                train_start_time = time.time()
                trainer.fit(train_values, train_labels, train_missing, mean, std)
                # 记录训练结束时间
                train_end_time = time.time()
                train_time = train_end_time - train_start_time
                
                # 记录测试开始时间
                test_start_time = time.time()
                test_score = predictor.get_score(test_values, test_missing)
                # 记录测试结束时间
                test_end_time = time.time()
                test_time = test_end_time - test_start_time
                
                # 由于 test_score 的长度与 test_labels 不同，需要对齐
                # test_score 的长度是 len(test_values) - x_dims + 1
                aligned_test_labels = test_labels[model.x_dims - 1:]

                # 计算评估指标 - 遍历不同阈值找到最佳F1分数
                best_f1 = 0
                best_threshold = 0
                best_precision = 0
                best_recall = 0
                
                # 遍历1到99的百分位数
                for percentile in range(1, 100):
                    threshold = np.percentile(test_score, percentile)
                    pred_labels = (test_score < threshold).astype(int)
                    #testscore过小
                    # 计算评估指标
                    precision = precision_score(aligned_test_labels, pred_labels, zero_division=0)
                    recall = recall_score(aligned_test_labels, pred_labels, zero_division=0)
                    f1 = f1_score(aligned_test_labels, pred_labels, zero_division=0)
                    
                    # 更新最佳结果
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                        best_precision = precision
                        best_recall = recall
                
                # 获取文件名（不含路径和扩展名）
                file_name = os.path.basename(csv_file_path).replace('.csv', '')
                
                # 将结果添加到本轮列表
                result_entry = {
                    'round': i+1,
                    'file_name': file_name,
                    'precision': best_precision,
                    'recall': best_recall,
                    'f1_score': best_f1,
                    'train_time': train_time,
                    'test_time': test_time
                }
                round_results.append(result_entry)
                results_summary.append(result_entry)
                
                print(f"File: {file_name}")
                print(f"Best Precision: {best_precision:.4f}")
                print(f"Best Recall: {best_recall:.4f}")
                print(f"Best F1-score: {best_f1:.4f}")
                print(f"Training time: {train_time:.2f} seconds")
                print(f"Testing time: {test_time:.2f} seconds")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error processing {csv_file_path}: {str(e)}")
            file_name = os.path.basename(csv_file_path).replace('.csv', '')
            result_entry = {
                'round': i+1,
                'file_name': file_name,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'train_time': 0,
                'test_time': 0
            }
            round_results.append(result_entry)
            results_summary.append(result_entry)

        # 保存本轮结果
        round_results_df = pd.DataFrame(round_results)
        if i == 0:
            # 第一轮创建新文件（包含表头）
            round_results_df.to_csv('results_summary1.csv', index=False,mode='a', header=True)
        else:
            # 后续轮次追加（不包含表头）
            round_results_df.to_csv('results_summary1.csv', index=False, mode='a', header=False)
        # 打印本轮汇总结果
        print(f"\n第 {i+1} 轮汇总结果:")
        round_df = pd.DataFrame(round_results)
        print(round_df.to_string(index=False))

# 循环结束后保存最终结果
results_df = pd.DataFrame(results_summary)


# 打印最终汇总结果
print("\n所有轮次的汇总结果:")
print(results_df.to_string(index=False))

# 按轮次统计结果
print("\n按轮次统计结果:")
for i in range(1, 6):
    round_data = results_df[results_df['round'] == i]
    if len(round_data) > 0:
        print(f"\n第 {i} 轮:")
        print(f"  处理文件数: {len(round_data)}")
        print(f"  平均 Precision: {round_data['precision'].mean():.4f}")
        print(f"  平均 Recall: {round_data['recall'].mean():.4f}")
        print(f"  平均 F1-score: {round_data['f1_score'].mean():.4f}")
        print(f"  平均训练时间: {round_data['train_time'].mean():.2f} 秒")
        print(f"  平均测试时间: {round_data['test_time'].mean():.2f} 秒")

# 总体统计
if len(results_df) > 0:
    print("\n总体统计:")
    print(f"  总处理文件数: {len(results_df)}")
    print(f"  平均 Precision: {results_df['precision'].mean():.4f}")
    print(f"  平均 Recall: {results_df['recall'].mean():.4f}")
    print(f"  平均 F1-score: {results_df['f1_score'].mean():.4f}")
    print(f"  平均训练时间: {results_df['train_time'].mean():.2f} 秒")
    print(f"  平均测试时间: {results_df['test_time'].mean():.2f} 秒")
    print(f"  总处理时间: {results_df['train_time'].sum() + results_df['test_time'].sum():.2f} 秒")