
import pandas as pd
import datetime
import torch
import math
import os
from base.loss_transfer import TransferLoss

def TDC(num_domain, data, dis_type='coral',dis=2):
    start_time = datetime.datetime.strptime(
        '2019-01-04 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(
        '2023-07-21 0:00:00', '%Y-%m-%d %H:%M:%S')
    # num_day = (end_time - start_time).days
    num_day=data.shape[0]
    split_N = 20
    data = data
    feat = data[0:num_day]
    feat = torch.tensor(feat.values, dtype=torch.float32)
    feat_shape_1 = feat.shape[1]
    feat = feat.reshape(-1, feat.shape[1])
    feat = feat.cuda()
    # num_day_new = feat.shape[0]

    selected = [0, 20]
    #生成从1到9的列表
    candidate = [i for i in range(2, 20, dis)]
    # candidate = [ 4,5,6,  9,10,14,16,18,19]
    start = 0
    if num_domain in [2, 3, 5, 7, 10]:
        while len(selected) - 2 < num_domain - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i - 1] / split_N * num_day) * feat_shape_1
                        index_part1_end = start + math.floor(selected[i] / split_N * num_day) * feat_shape_1
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_N * num_day) * feat_shape_1
                        index_part2_end = start + math.floor(selected[j + 1] / split_N * num_day) * feat_shape_1
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])
        selected.sort()
        res = []
        for i in range(1, len(selected)):
            if i == 1:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i - 1]), hours=0)
            else:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i - 1]) + 1,
                                                                 hours=0)
            sel_end_time = start_time + datetime.timedelta(days=int(num_day / split_N * selected[i]))
            sel_start_time = datetime.datetime.strftime(sel_start_time, '%Y-%m-%d %H:%M')
            sel_end_time = datetime.datetime.strftime(sel_end_time, '%Y-%m-%d %H:%M')
            res.append((sel_start_time, sel_end_time))
        return res
    else:
        print("error in number of domain")


def get_split_time(num_domain=2, mode='pre_process', data=None, dis_type='coral',dis=2):
    spilt_time = {
        '2': [('2019-1-4 0:0', '2019-6-30 23:0'), ('2019-7-1 0:0', '2023-10-28 23:0')]
    }
    if mode == 'pre_process':
        return spilt_time[str(num_domain)]
    if mode == 'tdc':
        return TDC(num_domain, data, dis_type=dis_type,dis=dis)
    else:
        print("error in mode")
def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i + 1, num_domain + 1):
            index.append((i, j))
    return index

def split_data(data, split_time_list):
    data_list = []
    for i in range(len(split_time_list)):
        start_time = split_time_list[i][0]
        end_time = split_time_list[i][1]
        data_list.append(data.loc[start_time:end_time].values.reshape(-1))
    return data_list