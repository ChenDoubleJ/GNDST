import os
import json
import random


def feature_choosing(example):
    X = []
    for epo in example:
        x_new = []
        for sa in epo:
            x2 = []
            for s in sa:
                x2.append(s[2:6] + [s[9]] + [s[-2]] + [s[-1]])  # 提取特征：卫星索引 仰角、方位角、载噪比、伪距、标签，系统标记
            x_new.append(x2)
        X.append(x_new)
    return X


# 读文件
def readfile(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()
    set = []
    epoch = []
    for line in lines[1:]:
        if line.split('\n')[0].split(',') != ['']:
            epoch.append(line.split('\n')[0].split(','))
        else:
            set.append(epoch)
            epoch = []
    return set


def stringTofloat(string_list):
    float_list = []
    for a in string_list:
        s = []
        for b in a:
            s1 = []
            for bb in b:
                s1.append([float(i) for i in bb])
            s.append(s1)
        float_list.append(s)
    return float_list


# 合并三个卫星系统在同一时刻的信息
def verification_merge(list1, list2, list3):
    final = []
    for epo_bd in list1:
        for epo_gps in list2:
            if epo_bd[0][1] == epo_gps[0][1]:
                for epo_gal in list3:
                    if epo_bd[0][1] == epo_gps[0][1] == epo_gal[0][1]:
                        # 标记系统属性
                        nepo_bd, nepo_gps, nepo_gal = [], [], []
                        for bds in epo_bd:
                            nbds = bds + [0.0]
                            nepo_bd.append(nbds)
                        for gps in epo_gps:
                            ngps = gps + [1.0]
                            nepo_gps.append(ngps)
                        for gal in epo_gal:
                            ngal = gal + [2.0]
                            nepo_gal.append(ngal)
                        final.append(nepo_bd + nepo_gps + nepo_gal)
                        break
                break
    return final


num = [0] * 22
ls = list(range(1, 22))
max = 0
sum = 0
# 读取数据
file_bd = os.listdir("data/BD/")
file_gps = os.listdir("data/GPS/")
file_gal = os.listdir("data/GAL/")
dataset = []
# print(file_gps)
for fn in range(1, len(file_bd) + 1):
    # print(fn)
    # date = fn.split('_')[0]
    list_bd = readfile(f"data/BD/BD{fn}.csv")
    list_gps = readfile(f"data/GPS/GPS{fn}.csv")  # "E:\\ZKG\\GPS\\"+date+"_B.csv"
    list_gal = readfile(f"data/GAL/GAL{fn}.csv")
    # print(len(list_bd), len(list_gps), len(list_gal))
    # print(list_b
    # d[0])
    multiple = verification_merge(list_bd, list_gps, list_gal)
    # #print(len(multiple))
    dataset.append(multiple)  # 每一个位置的每个时刻的卫星
for i in range(27):
    for j in range(len(dataset[i])):
        if len(dataset[i][j]) > max:
            max = len(dataset[i][j])
        num[len(dataset[i][j])] = num[len(dataset[i][j])] + 1
# # print(max)
print(ls)
print(num[1:])  # 每个时刻的卫星个数分布
dataset = feature_choosing(dataset)  # 把特征选出来
# print(len(dataset[0][80]))
# print(dataset[0][80])
dataset = stringTofloat(dataset)
# print(dataset[0][0])
# print(dataset[0][1])
# print(dataset[0][2])
train_data = []
# valid_data = []
test_data = []
T = 8  # 窗口大小c
# 泛化性测试

ran = [24, 11, 4, 15, 16]

# ran = [15, 16, 7]

print(len(dataset))
for i in range(len(dataset)):
    data = dataset[i]
    if i in ran:
        test_data.append(data)
    else:
        train_data.append(data)
# for data in dataset[1:]:
#     train_data.append(data)
# for data in dataset[:1]:
#     test_data.append(data)

# for data in dataset:
#     train_data.append(data[0:int(0.9 * len(data))])
#     test_data.append(data[int(0.9 * len(data)):])
train = []
test = []
for i in train_data:
    for j in range(0, (len(i) // T) * T, T):
        train.append(i[j:j + T])
print(len(train))
for i in test_data:
    for j in range(0, (len(i) // T) * T, T):
        test.append(i[j:j + T])
print(len(test))


with open('data/train.json', 'w') as file:
    json.dump(train, file)
with open('data/test.json', 'w') as file:
    json.dump(test, file)
