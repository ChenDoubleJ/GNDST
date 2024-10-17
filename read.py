
import csv
import numpy
import re
import os


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


def filtering(x):
    new_x = []
    for epoch in x:
        new_epoch = []
        for sa in epoch:
            if abs(float(sa[9])) < 100:
                new_epoch.append(sa)
        new_x.append(new_epoch)
    return new_x


def scaling(x):
    new = []
    for epoch in x:
        new_e = []
        for sa in epoch:
            ss = []
            ss.append(sa[0]/90)
            ss.append(sa[1]/360)
            ss.append(sa[2]/60)
            ss.append((sa[3]+100)/200)
            #ss.append((sa[4]-min)/(max-min))
            new_e.append(ss)
        new.append(new_e)
    return new


def time_seria(dataset):
    n, m = 25, 10
    padding = [[0] * n] * m

    new_dataset = []
    history = []
    new_labe = []
    new_error = []
    for data in dataset:
        satellite = {}
        new_data = []
        history_data = []
        data_label = []
        data_error = []
        ti = 0
        for epoch in data:
            ti += 1
            new_epoch = []
            history_epoch = []
            epoch_label = []
            epoch_error = []
            add_signal = False

            satellite_list_current = []
            for sa in epoch:
                satellite_list_current.append(sa[2])
            if ti > 1:
                for name in satellite_list_last:
                    if name not in satellite_list_current:
                        satellite.pop(name)

            for sa in epoch:
                new_epoch.append(sa)
                #print(sa)
                if sa[2] in satellite:
                    satellite[sa[2]].append(sa)
                else:
                    satellite[sa[2]] = [sa]
                if len(satellite[sa[2]]) > 10:
                    add_signal = True
                    history_epoch.append(satellite[sa[2]][-10:])
                    epoch_label.append([sa[-1]])
                    epoch_error.append([sa[6]])
                else:
                    history_epoch.append(padding)
                    epoch_label.append([-1])
                    epoch_error.append([0])
            if add_signal:
                new_data.append(new_epoch)
                history_data.append(history_epoch)
                data_label.append(epoch_label)
                data_error.append(epoch_error)

            satellite_list_last = []
            for sa in epoch:
                satellite_list_last.append(sa[2])

        new_dataset.append(new_data)
        history.append(history_data)
        new_labe.append(data_label)
        new_error.append(data_error)

    return new_dataset, history, new_labe, new_error


def get_epoch_features_float(dataset):
    new_dataset = []
    for data in dataset:
        new_data = []
        for epoch in data:
            new_epoch = []
            rss = 0
            for sa in epoch:
                rss += float(sa[9])**2
            for sa in epoch:
                new_epoch.append([float(sa[3])/90, float(sa[4])/360, float(sa[5])/60, (float(sa[9])+100)/200])
            new_data.append(new_epoch)
        new_dataset.append(new_data)
    return new_dataset


def get_history_features_float(history):
    new_history = []
    for his in history:
        new_his = []
        for epoch in his:
            new_epoch = []
            for sa in epoch:
                new_sa = []
                rss = 0
                for t in sa:
                    rss += float(t[9]) ** 2
                for t in sa:
                    #print(t)
                    new_sa.append([float(t[3])/90, float(t[4])/360, float(t[5])/60, (float(t[9])+100)/200])
                new_epoch.append(new_sa)
            new_his.append(new_epoch)
        new_history.append(new_his)
    return new_history


def get_label_float(labels):
    new_labels = []
    for label in labels:
        new_label = []
        for epoch in label:
            new_epoch = []
            for sa in epoch:
                new_epoch.append([float(sa[0])])
            new_label.append(new_epoch)
        new_labels.append(new_label)
    return new_labels


def add_bd_name(list_bd):
    for i in range(len(list_bd)):
        for j in range(len(list_bd[i])):
            list_bd[i][j][2] = 'C' + list_bd[i][j][2]
    return list_bd


def add_gps_name(list_bd):
    for i in range(len(list_bd)):
        for j in range(len(list_bd[i])):
            list_bd[i][j][2] = 'G' + list_bd[i][j][2]
    return list_bd


def add_gal_name(list_bd):
    for i in range(len(list_bd)):
        for j in range(len(list_bd[i])):
            list_bd[i][j][2] = 'E' + list_bd[i][j][2]
    return list_bd


def verification_merge(list1, list2, list3):
    final = []
    for epo_bd in list1:
        for epo_gps in list2:
            if epo_bd[0][1] == epo_gps[0][1]:
                for epo_gal in list3:
                    if epo_bd[0][1] == epo_gps[0][1] == epo_gal[0][1]:
                        final.append(epo_bd+epo_gps+epo_gal)
                        break
                break
    return final


def feature_choosing(example):
    X = []
    for epo in example:
        x_new = []
        if len(epo) > 0:
            for sa in epo:
                x_new.append(sa[3:7]+[sa[9]]+[sa[-1]])
            X.append(x_new)
    return X


def get_input(X_):
    input = []
    Y_v = []
    Y_p = []
    for x in X_:
        i = []
        v = []
        p = []
        rss = 0.0
        for s in x:
           rss += s[-2]**2
        for ss in x:
            j = []
            j.append(ss[0])
            j.append(ss[1])
            j.append(ss[2])
            j.append(ss[-2])
            j.append(rss)
            if int(ss[-1]) == 1:
               v.append([0])  # classification label
            elif int(ss[-1]) == 0:
               v.append([1])
            p.append([ss[-3]])  # errors regression
            i.append(j)
        Y_v.append(v)
        Y_p.append(p)
        input.append(i)
    return input, Y_v, Y_p


# def stringTofloat(string_list):
#     float_list = []
#     for a in string_list:
#         s = []
#         for b in a:
#             s.append([float(i) for i in b])
#         float_list.append(s)
#     return float_list



# file_bd = os.listdir("E:\\Model\\code\\BD\\")
# file_gps = os.listdir("E:\\Model\\code\\GPS\\")
# file_gal = os.listdir("E:\\Model\\code\\GAL\\")
#
# dataset = []
# number = 0
# for fn in file_bd[0:1]:
#     date = fn.split('_')[0]
#     list_bd = readfile("E:\\Model\\code\\BD\\"+fn)
#     list_gps = readfile("E:\\Model\\code\\GPS\\"+date+"_B.csv")
#     list_gal = readfile("E:\\Model\\code\\GAL\\"+fn)
#     print(len(list_bd), len(list_gps), len(list_gal))
#     multiple = verification_merge(list_bd, list_gps, list_gal)
#     print(len(multiple))
#     dataset.append(multiple)

# X__ = feature_choosing(dataset[0])  # extract features
# X_1 = stringTofloat(X__)  # convert format
# X_ = X_1[0:1000]
#
# X, Y1, Y2 = get_input(X_)  # generation input
#
# print(X[0])
# print(Y1[0])
# print(Y2[0])

