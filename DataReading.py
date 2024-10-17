# this is for data preparation
import numpy as np
import re


# LOS signal processing=========================

# # file path
# file_path = 'BDSRes.txt'
#
# # load txt
# f = open(file_path, 'r')
# data = f.readlines()
# # print original examples
# print(data[0].split(" "))
#
# # remove blank space and \n
# print(re.split(r'[ ]+', data[0][0:-1]))
#
#
# feature_str = []
# for line in data[1:]:
#     feature_str.append(re.split(r'[ ]+', line[0:-2]))
# print(feature_str[0])  # check features


def readtail2(length_tail, start_index, list):
    group = []
    group.append(list[start_index])
    for i in range(length_tail):
       if start_index+i+1 > len(list)-1:
           break
       elif list[start_index][3] == list[start_index + i + 1][3] and list[start_index][4] == list[start_index + i + 1][4]:
           group.append(list[start_index + i + 1])
       else:
           return group, i+1
    return group, i+1



def readtail(stride, start_index, list):
    group = []
    group.append(list[start_index])
    for i in range(stride):
       if start_index+i+1 > len(list)-1:
           break
       elif list[start_index][0] == list[start_index + i + 1][0] and list[start_index][1] == list[start_index + i + 1][1]:
           group.append(list[start_index + i + 1])
       else:
           return group, i+1
    return group, i+1


def stringTofloat(string_list):
    float_list = []
    for a in string_list:
        s = []
        for b in a:
            s.append([float(i) for i in b])
        float_list.append(s)
    return float_list


# example = []
# initial = 0
# for index in range(len(feature_str)):
#     epoch, i = readtail(50, initial, feature_str)
#     example.append(epoch)
#     initial += i
#     if initial > len(feature_str)-1:
#         break
#
# X = []

def featuresDecision(example):
    X = []
    for epo in example:
        x_new = []
        #print('************** ', len(epo))
        if len(epo) > 0:
            for sa in epo:
                fea = sa[4:]

                # fea.append(sa[5])
                # fea.append(sa[6])
                # fea.append(sa[7])
                # fea.append(sa[8])
                # fea.append(sa[-1])
                x_new.append(fea)
            X.append(x_new)
    return X

def featuresDecision2(example):
    X = []
    for epo in example:
        x_new = []
        if len(epo) > 0:
            for sa in epo:
                fea = sa[0:2] + sa[4:]

                # fea.append(sa[5])
                # fea.append(sa[6])
                # fea.append(sa[7])
                # fea.append(sa[8])
                # fea.append(sa[-1])
                x_new.append(fea)
            X.append(x_new)
    return X



def inputGeneration(X_):
    input = []
    Y_v = []
    Y_p = []
    for x in X_:
        i = []
        v = []
        p = []
        rss = 0.0
        for s in x:
           rss += s[-1]**2
        for ss in x:
            j = []
            j.append(ss[0])
            j.append(ss[1])
            j.append(ss[2])
            j.append(ss[-1])
            j.append(rss)
            if abs(ss[-1]) < 2.5:
                v.append([1])
            else:
                v.append([0])
            p.append([ss[3]])
            i.append(j)
        Y_v.append(v)
        Y_p.append(p)
        input.append(i)
    return input, Y_v, Y_p





# X_ = stringTofloat(X)


# def inputGeneration(X_):
#     input = []
#     Y_v = []
#     Y_p = []
#     for x in X_:
#         i = []
#         v = []
#         p = []
#         rss = 0.0
#         for s in x:
#            rss += s[-1]**2
#         for ss in x:
#             j = []
#             j.append(ss[0])
#             j.append(ss[1])
#             j.append(ss[2])
#             j.append(ss[-1])
#             j.append(rss)
#             if abs(ss[-1]) < 2.5:
#                 v.append([1])
#             else:
#                 v.append([0])
#             p.append([ss[3]])
#             i.append(j)
#         Y_v.append(v)
#         Y_p.append(p)
#         input.append(i)
#     return input, Y_v, Y_p

# input, Y_v, Y_p = inputGeneration(X_)

# print(input[5])
# print(Y_v[5])
# print(Y_p[5])
#
# print(len(input[5]))
# print(len(Y_v[5]))
# print(len(Y_p[5]))


# feature = np.array(feature_str, dtype=float)  # transform list to array
# print(feature[0])  # show example
# print(feature.dtype)  # check dtype
# print("Total number of examples: ", len(feature))  # number of examples

#
# # calculate RMS
# MS_phase = 0
# MS_code = 0
#
# print(feature[:, -1][2])  # print Res_code
# print(feature[:, -2][2])  # print Res_phase
#
# for i in range(len(feature[:, -1])):
#     MS_code += (feature[:, -1][i]) ** 2
#     MS_phase += (feature[:, -2][i]) ** 2
#
# assert len(feature[:, -1]) == len(feature[:, -2])
# MS_code = MS_code / len(feature[:, -1])
# MS_phase = MS_phase / len(feature[:, -2])
# RMS_code = MS_code ** 0.5
# RMS_phase = MS_phase ** 0.5
# print("the RMS of Res_code: ", RMS_code)
# print("the RMS of Res_phase: ", RMS_phase)
#
# clear_data = []
#
# # estimate signal
# for index in range(len(feature)):
#     if feature[index][3] > 35.0 and abs(feature[index][-1]) < RMS_code:
#         clear_data.append(feature[index])
# print(type(clear_data))
#
#
# # limited length
# clear_data_ = []
# for a in range(len(clear_data)):
#     s = []
#     for b in range(len(clear_data[a])):
#         s.append(round(clear_data[a][b], 4))
#     clear_data_.append(s)
#
#
# clear_data_list = []
# # transform list to string type
# for cl in clear_data_:
#     clear_data_list.append(list(map(str, cl)))
#
# print('The number of examples: ', len(clear_data_list))
#
# file_handle = open('BDS_LOS.txt', 'w')
# title = ['RovPrn ', 'Ele ', 'AZI ', 'SNR ', 'DDCode(m) ', 'DDPhase(m) ', 'ResPhase(m) ', 'ResCode(m)']
# file_handle.writelines(title)
# file_handle.writelines('\n')
#
# for line in range(len(clear_data_list)):
#     for i in range(len(clear_data_list[line])):
#         file_handle.write(clear_data_list[line][i])
#         file_handle.write(' ')
#     file_handle.write('\n')
# file_handle.close()

# =======================================================
