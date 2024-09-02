# 作者:     wxf

# 开发时间: 2023/11/8 18:51

# 需要输入：正样本、负样本

import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from torch import nn

# 自定义方法导入
from torch_sparse import tensor

from SigmoidKernel import SigmoidKernelDisease
from SigmoidKernel import SigmoidKernelRNA
from SnakeNNdisease import DSConvdisease
from SnakeNNcirc import DSConvcirc

# 方法定义
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

# Sigmoid函数
## 读取源文件 circ-mi的关系
OriginalData = []
ReadMyCsv(OriginalData, "648-circRNA-cancer-序号.csv")
print(len(OriginalData))
## 预处理
## 小写OriginalData
counter = 0
while counter < len(OriginalData):
    OriginalData[counter][0] = OriginalData[counter][0].lower()
    OriginalData[counter][1] = OriginalData[counter][1].lower()
    counter = counter + 1
# print('小写OriginalData')
AllSmall = []
counter = 0
while counter < len(OriginalData):
    Pair = []
    Pair.append(OriginalData[counter][0])
    Pair.append(OriginalData[counter][1])
    AllSmall.append(Pair)
    counter = counter + 1
# storFile(AllSmall, 'AllSmall.csv')
#LncDisease.csv将大写全部转化为小写
print('AllSmall长度', len(AllSmall))
print('OriginalData的长度', len(OriginalData))

# 构建AllMI
AllMI = []
counter1 = 0
while counter1 < len(OriginalData): #顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(AllMI):  #遍历AllDisease
        if OriginalData[counter1][0] != AllMI[counter2]:#有新疾病
            counter2 = counter2 + 1
        elif OriginalData[counter1][0] == AllMI[counter2]:#没有新疾病，用两个if第二个if会越界
            flag = 1
            break
    if flag == 0:
        AllMI.append(OriginalData[counter1][0])
    counter1 = counter1 + 1
print('len(AllMI)', len(AllMI))
# storFile(AllMI,'AllMI.csv')

# 构建AllCIRC
AllCIRC = []
counter1 = 0
while counter1 < len(OriginalData): #顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(AllCIRC):  #遍历AllDisease
        if OriginalData[counter1][1] != AllCIRC[counter2]:#有新疾病
            counter2 = counter2 + 1
        elif OriginalData[counter1][1] == AllCIRC[counter2]:#没有新疾病，用两个if第二个if会越界
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        AllCIRC.append(OriginalData[counter1][1])
    counter1 = counter1 + 1
print('len(AllCIRC)', len(AllCIRC))
# storFile(AllCIRC, 'AllCIRC.csv')

# 由drug-disease生成对应关系矩阵，有关系1，没关系0，行为疾病AllDisease，列为 AllDRUG
# 生成全0矩阵
MIAndCIRCBinary = []
counter = 0
while counter < len(AllCIRC):#行
    row = []
    counter1 = 0
    while counter1 < len(AllMI):#列
        row.append(0)
        counter1 = counter1 + 1
    MIAndCIRCBinary.append(row)
    counter = counter + 1
# print(MIAndCIRCBinary)

#将有关系对的进行重写，使得其在矩阵中值为1 剩余不变进行遍历矩阵
print('len(AllSmall)', len(AllSmall))
counter = 0
while counter < len(AllSmall):
    DN = AllSmall[counter][1]
    RN = AllSmall[counter][0]
    counter1 = 0
    while counter1 < len(AllCIRC):
        if AllCIRC[counter1] == DN:
            counter2 = 0
            while counter2 < len(AllMI):
                if AllMI[counter2] == RN:
                    MIAndCIRCBinary[counter1][counter2] = 1
                    break
                counter2 = counter2 + 1
            break
        counter1 = counter1 + 1
    counter = counter + 1
print('len(MIAndCIRCBinary)', len(MIAndCIRCBinary))
# storFile(MIAndCIRCBinary, 'MIAndCIRCBinary.csv')

matrix1 = SigmoidKernelDisease(MIAndCIRCBinary)
matrix2 = SigmoidKernelRNA(MIAndCIRCBinary)

storFile(matrix1, 'CDA_DIS_sigmoid.csv')
storFile(matrix2, 'CDA_CIRC_sigmoid.csv')

# 高斯核函数
# 读取源文件 circ-mi的关系
OriginalData = []
ReadMyCsv(OriginalData, "648-circRNA-cancer-序号.csv")
print(len(OriginalData))

# 预处理
# 小写OriginalData
counter = 0
while counter < len(OriginalData):
    OriginalData[counter][0] = OriginalData[counter][0].lower()
    OriginalData[counter][1] = OriginalData[counter][1].lower()
    counter = counter + 1
# print('小写OriginalData')
AllSmall = []
counter = 0
while counter < len(OriginalData):
    Pair = []
    Pair.append(OriginalData[counter][0])
    Pair.append(OriginalData[counter][1])
    AllSmall.append(Pair)
    counter = counter + 1
# storFile(AllSmall, 'AllSmall.csv')
#LncDisease.csv将大写全部转化为小写
print('AllSmall长度', len(AllSmall))
print('OriginalData的长度', len(OriginalData))

# 构建AllMI
AllMI = []
counter1 = 0
while counter1 < len(OriginalData): #顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(AllMI):  #遍历AllDisease
        if OriginalData[counter1][0] != AllMI[counter2]:#有新疾病
            counter2 = counter2 + 1
        elif OriginalData[counter1][0] == AllMI[counter2]:#没有新疾病，用两个if第二个if会越界
            flag = 1
            break
    if flag == 0:
        AllMI.append(OriginalData[counter1][0])
    counter1 = counter1 + 1
print('len(AllMI)', len(AllMI))
# storFile(AllMI,'AllMI.csv')



# 构建AllCIRC
AllCIRC = []
counter1 = 0
while counter1 < len(OriginalData): #顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(AllCIRC):  #遍历AllDisease
        if OriginalData[counter1][1] != AllCIRC[counter2]:#有新疾病
            counter2 = counter2 + 1
        elif OriginalData[counter1][1] == AllCIRC[counter2]:#没有新疾病，用两个if第二个if会越界
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        AllCIRC.append(OriginalData[counter1][1])
    counter1 = counter1 + 1
print('len(AllCIRC)', len(AllCIRC))
# storFile(AllCIRC, 'AllCIRC.csv')

# 由drug-disease生成对应关系矩阵，有关系1，没关系0，行为疾病AllDisease，列为 AllDRUG
# 生成全0矩阵
MIAndCIRCBinary = []
counter = 0
while counter < len(AllCIRC):#行
    row = []
    counter1 = 0
    while counter1 < len(AllMI):#列
        row.append(0)
        counter1 = counter1 + 1
    MIAndCIRCBinary.append(row)
    counter = counter + 1

# print(MIAndCIRCBinary)

#将有关系对的进行重写，使得其在矩阵中值为1 剩余不变进行遍历矩阵
print('len(AllSmall)', len(AllSmall))
counter = 0
while counter < len(AllSmall):
    DN = AllSmall[counter][1]
    RN = AllSmall[counter][0]
    counter1 = 0
    while counter1 < len(AllCIRC):
        if AllCIRC[counter1] == DN:
            counter2 = 0
            while counter2 < len(AllMI):
                if AllMI[counter2] == RN:
                    MIAndCIRCBinary[counter1][counter2] = 1
                    break
                counter2 = counter2 + 1
            break
        counter1 = counter1 + 1
    counter = counter + 1
print('len(MIAndCIRCBinary)', len(MIAndCIRCBinary))
# storFile(MIAndCIRCBinary, 'MIAndCIRCBinary.csv')

# 计算rd
counter1 = 0
sum1 = 0
while counter1 < (len(AllCIRC)):
    counter2 = 0
    while counter2 < (len(AllMI)):
        sum1 = sum1 + pow((MIAndCIRCBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nd = len(AllCIRC)
rdpie = 0.5
rd = rdpie * Nd / Ak
print('CIRC rd', rd)


# 生成CIRCGaussian
CIRCGaussian = []
counter1 = 0
while counter1 < len(AllCIRC):#计算疾病counter1和counter2之间的similarity
    counter2 = 0
    CIRCGaussianRow = []
    while counter2 < len(AllCIRC):# 计算Ai*和Bj*
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllMI):#疾病的每个属性分量
            sum2 = pow((MIAndCIRCBinary[counter1][counter3] - MIAndCIRCBinary[counter2][counter3]), 2)#计算平方
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj/rd))
        CIRCGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    CIRCGaussian.append(CIRCGaussianRow)
    counter1 = counter1 + 1
print('len(CIRCGaussian)', len(CIRCGaussian))
print('len(CIRCGaussian[0])', len(CIRCGaussian[0]))
storFile(CIRCGaussian, 'CDA_DIS_Guss.csv')


# disease circ
# drug mi
# 计算构建MIGaussian的rd
from numpy import *
MIAndCIRCBinary = np.array(MIAndCIRCBinary)    # 列表转为矩阵
MIAndCIRCBinary = MIAndCIRCBinary.T    # 转置DiseaseAndMiRNABinary
MIGaussian = []
counter1 = 0
sum1 = 0
while counter1 < (len(AllMI)):     # rna数量
    counter2 = 0
    while counter2 < (len(AllCIRC)):     # disease数量
        sum1 = sum1 + pow((MIAndCIRCBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nm = len(AllMI)
rdpie = 0.5
rd = rdpie * Nm / Ak
print('MI rd', rd)



# 生成MIGaussian
counter1 = 0
while counter1 < len(AllMI):   # 计算rna counter1和counter2之间的similarity
    counter2 = 0
    MIGaussianRow = []
    while counter2 < len(AllMI):   # 计算Ai*和Bj*
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllCIRC):   # rna的每个属性分量
            sum2 = pow((MIAndCIRCBinary[counter1][counter3] - MIAndCIRCBinary[counter2][counter3]), 2)#计算平方，有问题？？？？？
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj/rd))
        MIGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    MIGaussian.append(MIGaussianRow)
    counter1 = counter1 + 1
print('type(MIGaussian)', type(MIGaussian))
print('len(MIGaussian)', len(MIGaussian))
print('len(MIGaussian[0])', len(MIGaussian[0]))
storFile(MIGaussian, 'CDA_CIRC_Guss.csv')


# 矩阵合并，如果sigmoid为0则加入Guss

# 读取 CSV 文件并转换为 DataFrame
matrix1_df = pd.read_csv('CDA_DIS_sigmoid.csv', header=None)
matrix2_df = pd.read_csv('CDA_DIS_Guss.csv', header=None)

# 使用 numpy 数组处理 DataFrame
matrix1 = matrix1_df.values
matrix2 = matrix2_df.values
result1 = np.where(matrix1 > matrix2, matrix1, matrix2)
storFile(result1,"CDA_Matrix_mix_DIS.csv")

# 读取 CSV 文件并转换为 DataFrame
matrix3_df = pd.read_csv('CDA_CIRC_sigmoid.csv', header=None)
matrix4_df = pd.read_csv('CDA_CIRC_Guss.csv', header=None)

# 使用 numpy 数组处理 DataFrame
matrix3 = matrix3_df.values
matrix4 = matrix4_df.values
result2 = np.where(matrix3 > matrix4, matrix3, matrix4)
storFile(result2,"CDA_Matrix_mix_CIRC.csv")

# 原始的对角线矩阵
# 将对角线矩阵转置得到反对角线矩阵
anti_diagonal_matrix1 = np.fliplr(result1)
# print(anti_diagonal_matrix)
storFile(anti_diagonal_matrix1,"Anti_CDA_Matrix_mix_DIS.csv")

# 原始的对角线矩阵
# 将对角线矩阵转置得到反对角线矩阵
anti_diagonal_matrix2 = np.fliplr(result2)
# print(anti_diagonal_matrix)
storFile(anti_diagonal_matrix2,"Anti_CDA_Matrix_mix_CIRC.csv")

# 蛇形卷积disease
# 读取矩阵

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv0 = DSConvdisease(
        in_ch=1,
        out_ch=3,
        kernel_size=8,
        extend_scope=2,
        morph=1,
        if_offset=True,
        device=device)

# 从CSV文件中读取矩阵数据
matrix_data = np.loadtxt('Anti_CDA_Matrix_mix_DIS.csv', delimiter=',')
# 假设CSV文件中存储着形状为72x72的矩阵数据
# 将矩阵数据转换为卷积输入张量
A = torch.from_numpy(matrix_data).unsqueeze(0).unsqueeze(0)
# 使用torch.from_numpy()将NumPy数组转换为PyTorch张量
# 使用unsqueeze()方法在维度0和1上扩展张量的维度，以匹配目标张量大小

# 设置其他维度的大小
batch_size = 64
channel = 1
width = 72
height = 72

# 调整张量的维度大小
A = A.repeat(batch_size, channel, 1, 1)
A = A.view(batch_size, channel, width, height)

print('A的大小',A.size())  # 打印张量的大小，应为 [1, 1, 72, 72]
A = torch.from_numpy(matrix_data).unsqueeze(0).unsqueeze(0).float()
input_A_tensor = torch.Tensor(A)
out = conv0(input_A_tensor)
print('out形状',out.shape)
print('out类型', out.dtype)
output_np = np.squeeze(out.cpu().detach().numpy())
# 保存 NumPy 数组到 CSV 文件
# np.savetxt('output.csv', output_np.reshape(-1), delimiter=',')

# 找到每个位置最大值的通道索引
max_vals, max_indices = torch.max(out, dim=1)

# 将通道索引转换为72x72的矩阵
result = max_vals.detach().cpu().numpy().reshape(72, 72)
print(result.shape) # 输出结果的形状
print(result) # 输出转换后的矩阵
tensor_list = result.tolist()
storFile(tensor_list , 'DSconv_CDA_DIS.csv')


# 蛇形卷积circ
# 读取矩阵

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv1 = DSConvcirc(
        in_ch=1,
        out_ch=3,
        kernel_size=8,
        extend_scope=2,
        morph=1,
        if_offset=True,
        device=device)

# 从CSV文件中读取矩阵数据
matrix_data = np.loadtxt('Anti_CDA_Matrix_mix_CIRC.csv', delimiter=',')
# 假设CSV文件中存储着形状为72x72的矩阵数据
# 将矩阵数据转换为卷积输入张量
B = torch.from_numpy(matrix_data).unsqueeze(0).unsqueeze(0)
# 使用torch.from_numpy()将NumPy数组转换为PyTorch张量
# 使用unsqueeze()方法在维度0和1上扩展张量的维度，以匹配目标张量大小

# 设置其他维度的大小
batch_size = 100
channel = 1
width = 515
height = 515

# 调整张量的维度大小
B = B.repeat(batch_size, channel, 1, 1)
B = B.view(batch_size, channel, width, height)

print('B的大小',B.size())  # 打印张量的大小，应为 [1, 1, 72, 72]
B = torch.from_numpy(matrix_data).unsqueeze(0).unsqueeze(0).float()
input_B_tensor = torch.Tensor(B)
out1 = conv1(input_B_tensor)
print('out1形状',out1.shape)
print('out1类型', out1.dtype)
output_np1 = np.squeeze(out1.cpu().detach().numpy())
# 保存 NumPy 数组到 CSV 文件
# np.savetxt('output.csv', output_np.reshape(-1), delimiter=',')

# 找到每个位置最大值的通道索引
max_vals1, max_indices1 = torch.max(out1, dim=1)

# 将通道索引转换为72x72的矩阵
result1 = max_vals1.detach().cpu().numpy().reshape(515, 515)
print(result1.shape) # 输出结果的形状
print(result1) # 输出转换后的矩阵
tensor_list1 = result1.tolist()
storFile(tensor_list1 , 'DSconv_CDA_CIRC.csv')


# # 将矩阵转换为（1，64，64，1）的形状
# reshaped_matrix1 = np.expand_dims(matrix2, axis=(0, 3))
# print(reshaped_matrix1.shape)  # 打印转换后矩阵的形状
# A = reshaped_matrix1.astype(dtype = np.float32)
# A = torch.from_numpy(A)
# print('A形状',A.shape)
#
# if torch.cuda.is_available() :
#     A = A.to(device)
#     conv0 = conv0.to(device)
# out = conv0(A)
# print('out形状',out.shape)
# print('out类型', out.dtype)
#
# # 将数据转换为 numpy 数组
# reshaped_array2 = np.squeeze(out.cpu().detach().numpy())
# # 保存到 CSV 文件中
# np.savetxt('DSconv_circ.csv', reshaped_array2, delimiter=',')
#
# print('reshaped_array2形状', reshaped_array2.shape)
# print('reshaped_array2类型', reshaped_array2.dtype)



