import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data import DataLoader
# from torch_geometric.data import Data
# from torch_geometric.data import InMemoryDataset
# from torch_geometric.loader import DataLoader
from tqdm import tqdm


# 球面坐标转为三维坐标
def Angel_to_thr(el, az, r=1):
    h = np.abs(r * np.sin(el * np.pi / 180.))
    m = np.abs(r * np.cos(el * np.pi / 180.))
    x = m * np.cos(az * np.pi / 180.)
    y = m * np.sin(az * np.pi / 180.)
    return x, y, h


# 计算欧式距离
def Eus_dis(ve_mat):
    x = ve_mat[:, 0].reshape(-1, 1)
    y = ve_mat[:, 1].reshape(-1, 1)
    z = ve_mat[:, 2].reshape(-1, 1)
    xs = np.square(x - x.T)
    ys = np.square(y - y.T)
    zs = np.square(z - z.T)
    # dis_mat = np.sqrt(xs + ys + zs)
    dis_mat = np.arccos(1 - 0.5 * (xs + ys + zs))
    return dis_mat


# 计算球面距离
def Sphe_dis(ve_mat):
    x = ve_mat[:, 0].reshape(-1, 1)
    y = ve_mat[:, 1].reshape(-1, 1)
    z = ve_mat[:, 2].reshape(-1, 1)
    xs = x * x.T
    ys = y * y.T
    zs = z * z.T
    dis_mat = np.arccos(xs + ys + zs)
    return dis_mat


def find_num(mat):
    nmat = mat - np.eye(mat.shape[0])
    anum1 = np.where(nmat == 1)
    anum2 = np.dstack((anum1[0], anum1[1])).squeeze()
    return anum2.tolist()


def get_feature(train):
    temporal_feature = []
    spatial_feature = []
    labels = []
    for index in range(len(train)):  # 所有的窗口
        a1 = []  # 时序
        a2 = []  # 空间
        a3 = []  # lable
        # print(len(train[index]))  T
        for sa in range(len(train[index])):  # 每一个窗口T
            # print(len(train[index][sa]))  18
            b1 = []
            b2 = []
            for saa in range(len(train[index][sa])):  # 每一个时刻的卫星
                # print(len(train[index][sa][saa]))  # the number of features 7
                ss = train[index][sa][saa]
                j = []  # 时序
                j.append(ss[1])
                j.append(ss[2])
                j.append(ss[3])
                j.append(ss[4])
                if sa == len(train[index]) - 1:  # 每个窗口的最后一个时刻
                    i = []  # 空间
                    i.append(ss[1])
                    i.append(ss[2])
                    i.append(ss[3])
                    i.append(ss[4])
                    i.append(ss[6])  # 系统标记
                    a3.append(ss[-2])
                    a2.append(i)
                b1.append(j)
            a1.append(b1)
            # a2.append(b2)

        temporal_feature.append(a1)
        spatial_feature.append(a2)
        labels.append(a3)
    return temporal_feature, spatial_feature, labels


def get_input(X):
    edge_sets = []
    graph_fea = []
    dis = []
    sys = []
    for x in X:
        i = []
        sys_mark = []
        thrdim_list = []
        a = 0
        for ss in x:
            a += 1
            j = []
            j.append(ss[0])
            j.append(ss[1])
            j.append(ss[2])
            j.append(ss[3])
            i.append(j)

            # 获取系统标记
            sys_mark.append(ss[-1])
            # 获取三维坐标
            thrdim = Angel_to_thr(ss[0], ss[1], r=1)
            thrdim_list.append(thrdim)
        # print(a)
        # 系统属性边处理
        sys_npmark = np.array(sys_mark)
        nodes = np.array(sys_npmark)
        adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

        # 遍历每个结点，设置与同一系统的结点为1，对角线设置为0
        for f in range(len(nodes)):
            for j in range(len(nodes)):
                if f != j and nodes[f] == nodes[j]:
                    adjacency_matrix[f, j] = 1
        sys.append(adjacency_matrix)
        # sub_edge = find_num(sys_npmark == sys_npmark.T)
        # 邻近边处理
        thrdim_np = np.array(thrdim_list)
        # dis_mat = Sphe_dis(thrdim_np)
        dis_mat = Eus_dis(thrdim_np)
        dis.append(dis_mat)
        lin_edge = find_num((dis_mat < 0.7) & (dis_mat > 0.0))  # 阈值可以调节
        # for ed in lin_edge:
        #     if ed not in sub_edge:
        #         sub_edge.append(ed)
        sub_edge = lin_edge
        edge_sets.append(sub_edge)
        graph_fea.append(i)
        # print(graph_fea)
    return graph_fea, edge_sets, dis, sys


# def get_input(X_):

#     graph_fea = []
#     edge_sets = []
#     labels = []
#     a = 0
#     for x in X_:
#         i = []
#         v = []
#         p = []
#         sys_mark = []
#         thrdim_list = []
#         a += 1
#         for ss in x:
#             j = []
#             j.append(ss[0])
#             j.append(ss[1])
#             j.append(ss[2])
#             j.append(ss[3])
#             i.append(j)
#
#             v.append(ss[-2])
#             # 获取系统标记
#             sys_mark.append(ss[-1])
#             # 获取三维坐标
#             thrdim = Angel_to_thr(ss[0], ss[1], r=1)
#             thrdim_list.append(thrdim)
#
#         # 系统属性边处理
#         sys_npmark = np.array(sys_mark).reshape(1, -1)
#         sub_edge = find_num(sys_npmark == sys_npmark.T)
#         # 邻近边处理
#         thrdim_np = np.array(thrdim_list)
#         dis_mat = Sphe_dis(thrdim_np)
#         lin_edge = find_num(dis_mat < 0.9)
#         for ed in lin_edge:
#             if ed not in sub_edge:
#                 sub_edge.append(ed)
#         edge_sets.append(sub_edge)
#         labels.append(v)
#         graph_fea.append(i)
#     return graph_fea, edge_sets, labels


with open('data/dealed_trainset.json', 'r') as file:
    train = json.load(file)

with open('data/dealed_testset.json', 'r') as file:
    test = json.load(file)

print(f'The length of train is {len(train)}.')
print(f'The length of test is {len(test)}.')
# train = get_input(train)
temporal_feature1, spatial_feature1, labels1 = get_feature(train)  # 训练数据
temporal_feature2, spatial_feature2, labels2 = get_feature(test)  # 测试数据

# # 构建图数据的类
# class GNSSDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(GNSSDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_file_names(self):
#         return []
#
#     @property
#     def processed_file_names(self):
#         return ['GNSS.dataset']
#
#     def download(self):
#         pass
#
#     def process(self):
#         data_list = []
#         Node_fea = []
#         # Edge_sets = []
#         Labels = []
#         Node_fea.append(spatial_feature1)
#         Edge_sets = edge_fea1
#         Labels.append(labels1)
#         temporal_feature = temporal_feature1
#         for i in tqdm(range(len(Node_fea))):
#             node = torch.tensor(Node_fea[i])
#             edt = (torch.tensor(Edge_sets[i], dtype=torch.int64)).T
#             label = torch.LongTensor(Labels[i])
#             graph = Data(x=node, edge_index=edt, y=label)
#             data_list.append(graph)
#         data, slices = self.collate(data_list)
#         torch.save([data, slices], self.processed_paths[0])


spatial_feature1, edge_fea1, dis1, sys1 = get_input(spatial_feature1)
spatial_feature2, edge_fea2, dis2, sys2 = get_input(spatial_feature2)


# trainset = GNSSDataset(root='.datasets/data')
#
# train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
# for data in train_loader:
#     print(data.x.size())
#     print(data.edge_index.size())
#     break

# print(len(graph))
class DataSet(Dataset):
    def __init__(self, x, y, z, s, label):
        self.x = x
        self.y = y
        self.z = z
        self.s = s
        self.label = label

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.z[item], self.s[item], self.label[item]

    def __len__(self):
        return len(self.x)


length = len(edge_fea1)

for i in range(length):
    last = edge_fea1[i]
    last = np.array(last)
    last = last.T
    last = last.tolist()
    if i == 0:
        a = torch.tensor(last, dtype=torch.int64)
    else:
        last = torch.tensor(last, dtype=torch.int64)
        a = torch.cat((a, last), dim=1)

edge_fea1 = a
# print(edge_fea1.size())
length = len(edge_fea2)

for i in range(length):
    last = edge_fea2[i]
    last = np.array(last)
    last = last.T
    last = last.tolist()
    if i == 0:
        b = torch.tensor(last, dtype=torch.int64)
    else:
        last = torch.tensor(last, dtype=torch.int64)
        b = torch.cat((b, last), dim=1)
edge_fea2 = b
# print(edge_fea1.size())
# print(edge_fea2.size())

temporal_feature1 = torch.tensor(np.asarray(temporal_feature1).astype(np.float32))
spatial_feature1 = torch.tensor(np.asarray(spatial_feature1).astype(np.float32))
labels1 = torch.LongTensor(np.asarray(labels1))
temporal_feature2 = torch.tensor(np.asarray(temporal_feature2).astype(np.float32))
spatial_feature2 = torch.tensor(np.asarray(spatial_feature2).astype(np.float32))
labels2 = torch.LongTensor(np.asarray(labels2))
dis1 = torch.tensor(np.asarray(dis1).astype(np.float32))
dis2 = torch.tensor(np.asarray(dis2).astype(np.float32))
sys1 = torch.tensor(np.asarray(sys1).astype(np.float32))
sys2 = torch.tensor(np.asarray(sys2).astype(np.float32))
# edge_fea1 = torch.tensor(edge_fea1, dtype=torch.int64)
# edge_fea2 = torch.tensor(edge_fea2, dtype=torch.int64)

# print(edge_fea1.size())
# graph1 = Data(x=spatial_feature1, edge_index=edge_fea1, y=labels1)

data_train = DataSet(spatial_feature1, temporal_feature1, dis1, sys1, labels1)
data_test = DataSet(spatial_feature2, temporal_feature2, dis2, sys2, labels2)
torch.save(data_train, 'data/dataset_train')
torch.save(data_test, 'data/dataset_test')
data_train = torch.load('data/dataset_train')
train_loader = DataLoader(data_train, batch_size=32, shuffle=True)
# test_loader = DataLoader(data_test, batch_size=32, shuffle=True)
for i, (x, y, z, s, label) in enumerate(train_loader):
    print(x.size())
    print(y.size())
    print(z.size())
    print(s.size())
    #     print(z)
    #     print(label.shape)
    break
# for i, (x, y, label) in enumerate(test_loader):
#     print(x.shape)
#     print(y.shape)
#     print(label.shape)
# return data_train, data_test
# train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
# for data in train_loader:
#     print(data)
#     input = data.x.cuda()
#     print(input.size())
# inputs = torch.tensor(np.asarray(a).astype(np.float32))
# print(inputs.size())
# with open('data/temporal_feature.json', 'w') as file:  # 处理后的每个窗口的每个时刻的每个卫星的特征数据
#     json.dump(temporal_feature, file)
# with open('data/spatial_feature.json', 'w') as file:
#     json.dump(spatial_feature, file)
# with open('data/labels.json', 'w') as file:
#     json.dump(labels, file)
