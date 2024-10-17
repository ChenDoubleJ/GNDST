import json

"""
# 判断每个窗口的卫星可数 check
# 删除窗口中卫星数量较少的窗口 train_del  test_del
# 筛选窗口中的所有时刻卫星，删除不在两端位置出现的卫星  choose
# 在中间位置插入两端存在的卫星 sa_insert
# 进行扩充到最大的卫星数量，但是要保证记录卫星的星系和索引，应对后续构建图处理
"""


def check(train):
    for index in range(len(train)):
        tar = train[index]
        last = tar[-1]
        beg = tar[0]
        a = {}
        b = {}
        c = {}
        for i in last:
            if i[6] == 0.0:
                for j in beg:
                    if j[6] == 0.0 and j[0] == i[0]:
                        a[i[0]] = 0
                        break
            elif i[6] == 1.0:
                for j in beg:
                    if j[6] == 1.0 and j[0] == i[0]:
                        b[i[0]] = 0
                        break
            else:
                for j in beg:
                    if j[6] == 2.0 and j[0] == i[0]:
                        c[i[0]] = 0
                        break
    return a, b, c


# def sort(tar):
#     ls = [0] * 23
#     for i in tar:
#         ls[len(i)] = ls[len(i)] + 1
#     _max = max(ls)
#     max_ind = ls.index(_max)
#     return max_ind

def train_del(train):
    del_list = []
    for index in range(len(train)):
        # tar = train[index]
        last = train[index][-1]  # 存储在窗口的始终时刻都出现的卫星
        beg = train[index][0]
        a = {}  # 存储BD
        b = {}  # 存储GAL
        c = {}  # 存储GPS
        for i in last:
            if i[6] == 0.0:
                for j in beg:
                    if j[6] == 0.0 and j[0] == i[0]:
                        a[i[0]] = 0
                        break
            elif i[6] == 1.0:
                for j in beg:
                    if j[6] == 1.0 and j[0] == i[0]:
                        b[i[0]] = 0
                        break
            else:
                for j in beg:
                    if j[6] == 2.0 and j[0] == i[0]:
                        c[i[0]] = 0
                        break
        if (len(a) + len(b) + len(c)) > 18 or (len(a) + len(b) + len(c)) < 5:  # 删除卫星较少的窗口
            del_list.append(index)
            continue
    for ind in sorted(del_list, reverse=True):
        del train[ind]
    return train


def test_del(test):
    del_list = []
    for index in range(len(test)):
        # tar = train[index]
        last = test[index][-1]  # 存储在窗口的始终时刻都出现的卫星
        beg = test[index][0]
        a = {}  # 存储BD
        b = {}  # 存储GAL
        c = {}  # 存储GPS
        for i in last:
            if i[6] == 0.0:
                for j in beg:
                    if j[6] == 0.0 and j[0] == i[0]:
                        a[i[0]] = 0
                        break
            elif i[6] == 1.0:
                for j in beg:
                    if j[6] == 1.0 and j[0] == i[0]:
                        b[i[0]] = 0
                        break
            else:
                for j in beg:
                    if j[6] == 2.0 and j[0] == i[0]:
                        c[i[0]] = 0
                        break
        if (len(a) + len(b) + len(c)) > 18 or (len(a) + len(b) + len(c)) < 7:  # 删除卫星较少的窗口
            del_list.append(index)
            continue
    for ind in sorted(del_list, reverse=True):
        del test[ind]
    return test


# 筛选窗口中的所有时刻卫星，删除不在两端位置出现的卫星
def choose(train):
    for index in range(len(train)):
        # tar = train[index]
        last = train[index][-1]  # 存储在窗口的始终时刻都出现的卫星
        beg = train[index][0]
        a = {}  # 存储BD
        b = {}  # 存储GAL
        c = {}  # 存储GPS
        for i in last:
            if i[6] == 0.0:
                for j in beg:
                    if j[6] == 0.0 and j[0] == i[0]:
                        a[i[0]] = 0
                        break
            elif i[6] == 1.0:
                for j in beg:
                    if j[6] == 1.0 and j[0] == i[0]:
                        b[i[0]] = 0
                        break
            else:
                for j in beg:
                    if j[6] == 2.0 and j[0] == i[0]:
                        c[i[0]] = 0
                        break
        for sa in range(len(train[index])):  # 遍历每个窗口的每个时刻
            # print(len(train[index][sa]))  # 9
            sa_list = []
            for saa in range(len(train[index][sa])):  # 遍历每个时刻的每个卫星
                if train[index][sa][saa][6] == 0.0:
                    if train[index][sa][saa][0] not in a:
                        sa_list.append(saa)
                elif train[index][sa][saa][6] == 1.0:
                    if train[index][sa][saa][0] not in b:
                        sa_list.append(saa)
                else:
                    if train[index][sa][saa][0] not in c:
                        sa_list.append(saa)
            for t in sorted(sa_list, reverse=True):
                del train[index][sa][t]
    return train


# 在中间位置插入两端存在的卫星
def sa_insert(train):
    T = 8  # 窗口的大小
    for index in range(len(train)):  # 所有的窗口
        beg = train[index][0]  # 每一个窗口的第一个时刻
        last = train[index][T-1]
        # print(len(beg))
        # print(len(last)
        for sa in range(len(train[index][0])):  # 遍历第一个时刻的每一个卫星
            ind = train[index][0][sa][0]  # 卫星的标号
            ind_id = train[index][0][sa][6]  # 卫星的星系
            ind_label = train[index][T - 1][sa][5]  # 当前卫星的窗口的最后一个时刻的label
            f1 = train[index][0][sa][1]
            f2 = train[index][0][sa][2]
            f3 = train[index][0][sa][3]
            f4 = train[index][0][sa][4]
            f11 = train[index][T-1][sa][1]
            f22 = train[index][T-1][sa][2]
            f33 = train[index][T-1][sa][3]
            f44 = train[index][T-1][sa][4]
            # print(ind)
            # print(ind_label)
            for saa in range(1, T-1):
                append = []
                w1 = (T - saa) / T
                w2 = saa / T
                if sa < len(train[index][saa]):  # 每个时刻的卫星的个数，防止out of list
                    if train[index][saa][sa][0] == ind and train[index][saa][sa][6] == ind_id:  # 表示存在
                        continue
                    else:  # 不存在 进行插值
                        append.append(ind)
                        append.append(round(w1 * f1 + w2 * f11, 2))
                        append.append(round(w1 * f2 + w2 * f22, 2))
                        append.append(round(w1 * f3 + w2 * f33, 2))
                        append.append(round(w1 * f4 + w2 * f44, 4))
                        append.append(ind_label)
                        append.append(ind_id)
                        train[index][saa].insert(sa, append)
                else:  # 超出了 就直接进行.append
                    append.append(ind)
                    append.append(round(w1 * f1 + w2 * f11, 2))
                    append.append(round(w1 * f2 + w2 * f22, 2))
                    append.append(round(w1 * f3 + w2 * f33, 2))
                    append.append(round(w1 * f4 + w2 * f44, 4))
                    append.append(ind_label)
                    append.append(ind_id)
                    train[index][saa].insert(sa, append)
    return train

# 进行扩充到最大的卫星数量，但是要保证记录卫星的星系和索引，应对后续构建图处理
def data_padding(train, maxnum):
    T = 8
    for index in range(len(train)):
        length = len(train[index][0])
        pad_len = maxnum - length

        # print(padding_value)
        for sa in range(T):
            padding_value = train[index][sa][length - 1]
            for i in range(pad_len):
                train[index][sa].append(padding_value)
    return train

with open('data/train.json', 'r') as file:
    train = json.load(file)

with open('data/test.json', 'r') as file:
    test = json.load(file)

# print(len(train))
# print(len(test))
# print(len(train[0]))
# print(train[0])
# print(len(train[0][0]))
# 判断每个窗口的卫星可数 check
#####################################################
# 判断train和test的没个窗口中的卫星个数
# sum = [0] * 29
# for i in train:
#     a, b, c = check(i)
#     print(a, end="  ")
#     print(b, end="  ")
#     print(c)
#     print(len(a) + len(b) + len(c))
#     sum[len(a) + len(b) + len(c) - 1] = sum[len(a) + len(b) + len(
#         c) - 1] + 1  # [0, 1, 5, 18, 34, 55, 81, 130, 166, 222, 424, 553, 573, 487, 342, 309, 257, 141, 63, 2, 0, 0]
#     print()
# print(sum)
# for i in test:
#     a, b, c = check(i)
#     print(a, end="  ")
#     print(b, end="  ")
#     print(c)
#     print(len(a) + len(b) + len(c))
#     sum[len(a) + len(b) + len(c) - 1] = sum[len(a) + len(b) + len(
#         c) - 1] + 1  # [0, 0, 2, 2, 4, 14, 31, 27, 53, 76, 91, 125, 161, 109, 93, 67, 59, 38, 7, 1, 0, 0, 0]
#     print()
# print(sum)
##################################################################
# 删除窗口中卫星数量较少的窗口 train_del  test_del
train = train_del(train)  # 返回删选后的训练和测试机，并且返回
test = test_del(test)
print(f'The length of train is {len(train)}')  # 删除一些窗口中卫星数量较少的窗口后的卫星的训练集合测试集的个数
print(f'The length of train is {len(test)}')
# print(len(train[0]))
# ########################################################################
# # 筛选窗口中的所有时刻卫星，删除不在两端位置出现的卫星  choose
train = choose(train)
test = choose(test)  # 获得训练集合测试集的每个窗口的其实位置和结束位置的卫星是一样的，中间可能缺少一些卫星
# print(len(train[0]))
# train = sa_insert(train)
###############################################################
# print(len(train[0][0]))
# [][][]第一个窗口,的第一个时刻的所有卫星,每个时刻的所有卫星
# 在中间位置插入两端存在的卫星
train = sa_insert(train)
test = sa_insert(test)
# # 进行扩充到最大的卫星数量，但是要保证记录卫星的星系和索引，应对后续构建图处理 18个 先尝试使用一个卫星进行填充到最大的数量的卫星的个数
# # train = data_padding(train, 18)
train = data_padding(train, 18)
test = data_padding(test, 18)
## 判断训练数据和测试数据每个窗口的卫星个数的最大和最小值
tr1 = []
te1 = []
max_train = 0
max_test = 0
min_train = 100
min_test = 100
for i in train:
    tr2 = []
    for j in i:
        # print(len(j), end=" ")
        tr2.append(len(j))
        if len(j) > max_train:
            max_train = len(j)
        if len(j) < min_train:
            min_train = len(j)
    tr1.append(tr2)
    # print()
for i in test:
    te2 = []
    for j in i:
        # print(len(j), end=" ")
        te2.append(len(j))
        if len(j) > max_test:
            max_test = len(j)
        if len(j) < min_test:
            min_test = len(j)
    te1.append(te2)
    # print()
print(max_train)
print(max_test)
print(min_train)
print(min_test)
with open('data/len_train.json', 'w') as file:  # 处理后的每个窗口的每个时刻的卫星的个数
    json.dump(tr1, file)
with open('data/len_test.json', 'w') as file:
    json.dump(te1, file)
with open('data/dealed_trainset.json', 'w') as file:  # 处理后的每个窗口的每个时刻的每个卫星的特征数据
    json.dump(train, file)
with open('data/dealed_testset.json', 'w') as file:
    json.dump(test, file)

