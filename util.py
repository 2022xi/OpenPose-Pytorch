import cv2
import math
import numpy as np

def draw(config, input_image, coords, subset, resize_fac = 1):
    stickwidth = 4
    canvas = input_image.copy()
    white = 255 * np.ones(input_image.shape)
    # 以列表返回可遍历的(键, 值) 元组数组。
    for body_part_type, body_part_meta in config.body_parts.items():
        color = body_part_meta.color
        body_part_peaks = coords[body_part_type.name]

        for peak in body_part_peaks:
            a = peak[0] * resize_fac
            b = peak[1] * resize_fac
            # 绘制圆
            cv2.circle(canvas, (a, b), stickwidth, color, thickness=-1)
            # 在纯白背景上画
            cv2.circle(white, (a, b), stickwidth, color, thickness=-1)

    cur_white = white.copy()

    # dict(id: [x,y])
    xy_by_id = dict([(item[3], np.array([item[0], item[1]])) for sublist in coords.values() for item in sublist])

    xy = np.zeros((2,2))
    for i, conn_type in enumerate(config.connection_types):
        index1 = config.body_parts[conn_type.from_body_part].slot_idx
        index2 = config.body_parts[conn_type.to_body_part].slot_idx
        indexes = np.array([index1, index2])        
        for s in subset:

            ids = s[indexes]            
            if -1 in ids:
                continue

            cur_canvas = canvas.copy()
            xy[0, :] = xy_by_id[ids[0]]
            xy[1, :] = xy_by_id[ids[1]]

            m_x = np.mean(xy[:, 0])
            m_y = np.mean(xy[:, 1])

            length = ((xy[0, 0] - xy[1, 0]) ** 2 + (xy[0, 1] - xy[1, 1]) ** 2) ** 0.5
            # 计算向量中的角度
            angle = math.degrees(math.atan2(xy[0, 1] - xy[1, 1], xy[0, 0] - xy[1, 0]))

            polygon = cv2.ellipse2Poly((int(m_x * resize_fac), int(m_y * resize_fac)),
                                       (int(length * resize_fac / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, conn_type.color)
            # 在纯白背景上画
            cv2.fillConvexPoly(cur_white, polygon, conn_type.color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas, white, cur_white

def pad_right_down_corner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(
            weights_name.split('.')[1:])]
    return transfered_model_weights
