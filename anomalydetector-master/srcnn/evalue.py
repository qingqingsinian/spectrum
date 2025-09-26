import os
from srcnn.competition_metric import get_variance, evaluate_for_all_series
import time
import json
import argparse
from msanomalydetector.spectral_residual import SpectralResidual
from srcnn.utils import *
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def auto():
    path_auto = os.getcwd() + '/auto.json'
    with open(path_auto, 'r+') as f:
        store = json.load(f)
    window = store['window']
    epoch = store['epoch']
    return window, epoch


def getfid(path):
    return path.split('/')[-1]


def get_path(data_source):
    if data_source == 'kpi':
        dir_ = root + '/Test/'
        trainfiles = [dir_ + _ for _ in os.listdir(dir_)]
        files = trainfiles
    else:
        dir_ = root + '/' + data_source + '/'
        files = [dir_ + _ for _ in os.listdir(dir_)]
    return files


def get_score_batch(data_source, files, thres, option, batch_size=256):
    total_time = 0
    results = []
    savedscore = []
    batch_metrics = []  # 存储每个批次的评估指标
    
    for f in files:
        print('reading', f)
        if data_source == 'kpi' or data_source == 'test':
            in_timestamp, in_value, in_label = read_csv_kpi(f)
        else:
            tmp_data = read_pkl(f)
            in_timestamp, in_value, in_label = tmp_data['timestamp'], tmp_data['value'], tmp_data['label']
        
        length = len(in_timestamp)
        if model == 'sr_cnn' and len(in_value) < window:
            print("length is shorter than win_size", len(in_value), window)
            continue
            
        # 存储所有批次的结果
        all_timestamps = []
        all_labels = []
        all_predictions = []
        all_scores = []
        
        # 分批处理数据
        batch_times = []
        batch_precisions = []
        batch_recalls = []
        batch_f1s = []
        
        # 按照指定的批次大小处理数据
        for i in range(0, length, batch_size):
            batch_end = min(i + batch_size, length)
            batch_timestamp = in_timestamp[i:batch_end]
            batch_value = in_value[i:batch_end]
            batch_label = in_label[i:batch_end]
            
            # 如果批次数据少于窗口大小，跳过（因为模型需要足够的窗口数据）
            if len(batch_value) < window:
                print(f"Batch size {len(batch_value)} is shorter than window size {window}, skipping")
                continue
                
            # 对当前批次进行预测
            time_start = time.time()
            batch_ts, batch_lbl, batch_pred, batch_scores = models[model](
                batch_timestamp, batch_value, batch_label, window, net, option, thres)
            time_end = time.time()
            
            batch_time = time_end - time_start
            batch_times.append({
                'file': f,
                'batch_start': i,
                'batch_end': batch_end,
                'time': batch_time
            })
            
            # 存储结果
            all_timestamps.extend(batch_ts)
            all_labels.extend(batch_lbl)
            all_predictions.extend(batch_pred)
            all_scores.extend(batch_scores)
            # 计算当前批次的评估指标
            if len(set(batch_pred)) > 1 :  # 确保有足够的样本计算指标
                anomaly_positions = []
                anomaly_count = 0

                for i, pred in enumerate(batch_pred):
                    if pred == 1:  # 异常点
                        anomaly_positions.append(i)  # 位置索引
                        anomaly_count += 1

                print(f"异常点个数: {anomaly_count}")
                print(f"异常点位置: {anomaly_positions}")
            
            total_time += batch_time
        
        # 将整个文件的结果添加到总体结果中
        results.append([all_timestamps, all_labels, all_predictions, f])
        savedscore.append([all_labels, all_scores, f, all_timestamps])
    
    return total_time, results, savedscore, batch_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRCNN')
    parser.add_argument('--data', type=str, required=True, help='location of the data file')
    parser.add_argument('--window', type=int, default=128, help='window size')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='snapshot', help='model path')
    parser.add_argument('--delay', type=int, default=3, help='delay')
    parser.add_argument('--thres', type=float, default=0.95, help='initial threshold of SR')  # 改为float类型
    parser.add_argument('--auto', type=bool, default=False, help='Automatic filling parameters')
    parser.add_argument('--model', type=str, default='sr_cnn', help='model')
    parser.add_argument('--missing_option', type=str, default='anomaly',
                        help='missing data option, anomaly means treat missing data as anomaly')

    args = parser.parse_args()
    if args.auto:
        window, epoch = auto()
    else:
        window = args.window
        epoch = args.epoch
    data_source = args.data
    delay = args.delay
    model = args.model
    root = os.getcwd()
    print(data_source, window, epoch)  # 修复变量名
    
    models = {
        'sr_cnn': sr_cnn_eval,
    }

    model_path = root + '/' + args.model_path + '/srcnn_retry' + str(epoch) + '_' + str(window) + '.bin'
    srcnn_model = Anomaly(window)
    net = load_model(srcnn_model, model_path).cuda()
    files = get_path(data_source)
    
    # 使用新的批处理函数，每次处理256条数据
    total_time, results, savedscore, batch_metrics = get_score_batch(
        data_source, files, args.thres, args.missing_option, batch_size=256)
    
    print('\n***********************************************')
    print('data source:', data_source, '     model:', model)
    print('-------------------------------')
    total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(results, delay)
    with open(data_source + '_saved_scores.json', 'w') as f:
        json.dump(savedscore, f)
    print('Total time used for making predictions:', total_time, 'seconds')
    
    # 输出每个文件的批处理指标
    print('\nBatch Evaluation Metrics:')
    print('-------------------------------')
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_times = []
    
    for metric in batch_metrics:
        print(f"File: {metric['file']}")
        print(f"  Avg Precision: {metric['avg_precision']:.4f}")
        print(f"  Avg Recall: {metric['avg_recall']:.4f}")
        print(f"  Avg F1-Score: {metric['avg_f1']:.4f}")
        print(f"  Batch Count: {metric['batch_count']}")
        
        all_precisions.append(metric['avg_precision'])
        all_recalls.append(metric['avg_recall'])
        all_f1s.append(metric['avg_f1'])
        
        # 计算该文件的总处理时间
        file_total_time = sum([t['time'] for t in metric['batch_times']])
        all_times.append(file_total_time)
        print(f"  Total Processing Time: {file_total_time:.4f} seconds")
        
        # 输出前几个批次的时间详情
        print(f"  Batch Times (first 5):")
        for i, batch_time in enumerate(metric['batch_times'][:5]):
            print(f"    Batch {batch_time['batch_start']}-{batch_time['batch_end']}: {batch_time['time']:.4f}s")
        if len(metric['batch_times']) > 5:
            print(f"    ... and {len(metric['batch_times']) - 5} more batches")
        print()
    
    # 输出总体平均指标
    if all_precisions:
        print(f"Overall Average Metrics:")
        print(f"  Precision: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
        print(f"  Recall: {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
        print(f"  F1-Score: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
        print(f"  Average Processing Time per File: {np.mean(all_times):.4f} ± {np.std(all_times):.4f} seconds")

    # 保留原有的阈值优化代码
    best = 0.
    bestthre = 0.
    print('delay :', delay)
    if data_source == 'yahoo':
        sru = {}
        rf = open(data_source + 'sr3.json', 'r')
        srres = json.load(rf)
        for (srtime, srl, srpre, srf) in srres:
            sru[getfid(srf)] = [srtime, srl, srpre]
        for i in range(98):
            newresults = []
            threshold = 0.01 + i * 0.01
            for f, (srtt, srlt, srpret, srft), (flabel, cnnscores, cnnf, cnnt) in zip(files, srres, savedscore):
                fid = getfid(cnnf)
                srtime = sru[fid][0]
                srl = sru[fid][1]
                srpre = sru[fid][2]
                srtime = [(srtime[0] - 3600 * (64 - j)) for j in range(64)] + srtime
                srl = [0] * 64 + srl
                srpre = [0] * 64 + srpre
                print(len(srl), len(flabel), '!!')
                assert (len(srl) == len(flabel))
                pre = [1 if item > threshold else 0 for item in cnnscores]
                newresults.append([srtime, srpre, pre, f])
            total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(newresults, delay, prt=False)
            if total_fscore > best:
                best = total_fscore
                bestthre = threshold
        results = []
        threshold = bestthre
        print('guided threshold :', threshold)
        for f, (flabel, cnnscores, _, ftimestamp) in zip(files, savedscore):
            pre = [1 if item > threshold else 0 for item in cnnscores]
            results.append([ftimestamp, flabel, pre, f])
        print('score\n')
        total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(results, delay)
        print(total_fscore)
    best = 0.
    for i in range(98):
        newresults = []
        threshold = 0.01 + i * 0.01
        for f, (flabel, cnnscores, _, ftimestamp) in zip(files, savedscore):
            pre = [1 if item > threshold else 0 for item in cnnscores]
            newresults.append([ftimestamp, flabel, pre, f])
        total_fscore, pre, rec, TP, FP, TN, FN = evaluate_for_all_series(newresults, delay, prt=False)
        if total_fscore > best:
            best = total_fscore
            bestthre = threshold
            print('tem best', best, threshold)
    threshold = bestthre
    print('best overall threshold :', threshold, 'best score :', best)