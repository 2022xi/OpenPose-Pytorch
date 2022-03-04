import math
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import time
import config
import connections
import coordinates
import estimators
import util
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from models.rtpose_vgg import get_model
# from models.rtpose_vgg_all import get_model

model_path = 'models/half_pose.pth'  # full_pose.pth
model = get_model(trunk='vgg19')
# 当使用完整的模型参数时，需要注释掉这一行
model = torch.nn.DataParallel(model)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
model.float()

allTime = []
trainTime = []
for i in range(1,2):
    # 然后读取一张测试图片（视频fps较低，不流畅）
    begin = time.time()
    test_img = "%d.jpg" % i # Image path here
    oriImg = cv2.imread("demo/"+test_img) # B,G,R order

    inp_size = 368  # 模型的输入大小
    stride = 8  # 模型输出的map尺寸为49，368/49=8
    padValue = 128
    thre1 = 0.1
    thre2 = 0.05
    scale_factors = [0.5, 1, 1.5, 2]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    multipliers = [x * inp_size / oriImg.shape[0] for x in scale_factors]  # 不同尺度下的放缩系数

    for m, scale in enumerate(multipliers):
        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # pad_right_down_corner函数是为了使图像imageToTest尺寸为stride的倍数，不足的地方，在右和下用padValue填补
        imageToTest_padded, pad = util.pad_right_down_corner(imageToTest, stride, padValue)

        # required shape (1, width, height, channels) for TensorFlow
        # 除256再-0.5的操作是为了归一化
        # np.newaxis为了增加一个维度
        # Pytorch requires (1, channels, width, height)
        im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5


        data = torch.from_numpy(im).float()
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            Mconv7_stage6_L1, Mconv7_stage6_L2 = model(data)
        Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
        Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

        print("Output shape (heatmap): " + str(Mconv7_stage6_L2.shape))
        print("Output shape (paf): " + str(Mconv7_stage6_L1.shape))


        # extract outputs, resize, and remove padding
        # output 1 is heatmaps
        # np.squeeze从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))
        heatmap = cv2.resize(heatmap, (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]  # 去除填充
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # output 0 is PAFs
        paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))
        paf = cv2.resize(paf, (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multipliers)
        paf_avg = paf_avg + paf / len(multipliers)
    train_end = time.time()
    trainTime.append(train_end-begin)
    
    
    thre1 = 0.1
    thre2 = 0.05


    cfg = config.get_default_configuration()
    # 得到各个身体部分的坐标和得分情况，一共有18个人体关键点。(x ,y, score, id)
    coords = coordinates.get_coordinates(cfg, heatmap_avg, thre1)
    # 得到部分的连接关系，一共有19种。(id1, id2, score1, score2, total score)
    conns = connections.get_connections(cfg, coords, paf_avg, thre2)
    skeletons = estimators.estimate(cfg, conns)

    print('coordinates: ')
    print(len(coords),coords["left_eye"])
    print('connections: ')
    print(len(conns))
    print('skeletons: ')
    print(skeletons)

    canvas,white,cur_white = util.draw(cfg, oriImg, coords, skeletons)
    # 保存处理后的结果
    cv2.imwrite("result/%d_canvas.png" % i, canvas)
    end = time.time()
    allTime.append(end-begin)

with open("testTime.txt", "w") as f:
    print("allTime:",allTime, file = f)
    print("trainTime:",trainTime, file = f)