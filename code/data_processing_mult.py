"""
@Time : 2021/9/30 11:01
@Auth : zy
"""

# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/16 8:39
@Auth ： zy

"""
import random
import numpy as np
# list = ['AGCTT', 'ACCGT', 'AACGT']
# lable = np.array([[1],[0],[1]])
#
# Q = np.array(list)
import xlrd
def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)

    booksheet_pos = workbook.sheet_by_index(0)
    nrows_pos = booksheet_pos.nrows

    booksheet_neg = workbook.sheet_by_index(1)
    nrows_neg = booksheet_neg.nrows

    seq = []
    label = []
    for i in range(nrows_pos):
        seq.append(booksheet_pos.row_values(i)[0])
        label.append(booksheet_pos.row_values(i)[1])
    for j in range(nrows_neg):
        seq.append((booksheet_neg.row_values(j)[0]))
        label.append(booksheet_neg.row_values(j)[1])

    return seq, np.array(label).astype(int)

def ACGTto0123(filename):
    seq, label = read_seq_label(filename)
    seq0123 = []
    for i in range(len(seq)):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq0123.append(one_seq)
    return seq0123, label

def seq_to01_to0123(filename):

    seq, label = read_seq_label(filename)

    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq_start = 0
        for j in range(seq_len):
            seq_0123[i, j] = int(one_seq[j - seq_start])
            if j < seq_start:
                seq_01[i, j, :] = 0.25
            else:
                try:
                    seq_01[i, j, int(one_seq[j - seq_start])] = 1
                except:
                    seq_01[i, j, :] = 0.25
    return seq_01, seq_0123, label

def name_and_length(data_path):

    my_dict = {}
    with open(data_path, "r", encoding='UTF-8') as f:
        for line in f.readlines():

            this_key=''
            this_value_len=0

            if line.startswith('>'):
                this_key = line[0:-1]
            elif line.strip() == "":
                pass
            else:
                this_value_len = len(line)

            a=1





    a=1





# return seq_name, seq, seq_len, seq_cut
def read_test_txt(data_path):

    seq_name = []
    seq = []
    seq_len = []
    seq_cut = []

    with open(data_path, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            if line.startswith('>'):
                line = line[0:-1]
                seq_name.append(line)
            elif line.strip() == "":
                pass
            else:
                if line.endswith('\n'):
                    line = line[0:-1]
                seq.append(line)
                seq_len.append(len(line))

                for i in range(len(line)-40):
                    seq_cut.append(line[i:i+41])

    return seq_name, seq, seq_len, seq_cut





def read_test_txt_to01_to0123(data_path):

    seq_name, _, each_len, seq_cut = read_test_txt(data_path)



    nrows = len(seq_cut)
    seq_len = len(seq_cut[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq_cut[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        one_seq = one_seq.replace('U', '3')
        seq_start = 0
        for j in range(seq_len):
            seq_0123[i, j] = int(one_seq[j - seq_start])
            if j < seq_start:
                seq_01[i, j, :] = 0.25
            else:
                try:
                    seq_01[i, j, int(one_seq[j - seq_start])] = 1
                except:
                    seq_01[i, j, :] = 0.25
    return seq_name, each_len, seq_cut, seq_01


if __name__ == '__main__':

    data_path = 'D://zyD//00BS//P3web//zyMult_web_root//PredictDataOFUsers//20210422232234666.txt'
    # read_test_txt(data_path)
    read_test_txt_to01_to0123(data_path)





