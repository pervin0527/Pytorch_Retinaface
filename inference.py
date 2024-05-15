import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from models.retinaface import RetinaFace
from data import cfg_mnet, cfg_re50

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm


def retinaface_inf(test_img, model):
    img = np.float32(test_img)
    im_height, im_width, _ = img.shape

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = model(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    ## ignore low scores. confidence threshold보다 높은 확률을 가진 결과만 남김.
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    ## keep top-K before NMS. 결과를 정렬하고, topk개를 뽑아낸다.
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    ## do NMS. 하나의 객체를 대상으로 만들어진 여러 개의 박스 중 가장 적절한 박스 하나만 남기고 나머지는 모두 삭제
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]

    fps_ = round(1/(time.time() - tic), 2)
    for b in dets:
        if b[4] < vis_thres:
            continue
        b = list(map(int, b))
        cv2.rectangle(test_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
        cx = b[0]
        cy = b[1] + 12

        # cv2.putText(test_img, text=f'{b[4]:.2f}', org=(cx, cy), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(255, 255, 255))

    cv2.putText(test_img, "retinaface", (410,70),cv2.FONT_HERSHEY_DUPLEX, 1.5,(255,0,0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(test_img, "fps : "+str(fps_), (5,70),cv2.FONT_HERSHEY_DUPLEX, 1.5,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
    return test_img


weight_path = '/home/pervinco/Pytorch_Retinaface/weights/Resnet50_Final.pth'
cfg = cfg_re50 # mobile0.25 (cfg_mnet) or resnet50 (cfg_re50)
resize = 1
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
vis_thres = 0.6

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = RetinaFace(cfg, phase = 'test').to(device) 

# model.load_state_dict(torch.load(weight_path, map_location=device))
state_dict = torch.load(weight_path, map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)

model.eval()
print("Model Loaded!")

test_path = '/home/pervinco/Datasets/WIDER/WIDER_val/val/images/0--Parade/0_Parade_marchingband_1_20.jpg'
test_img = cv2.imread(test_path)

result = retinaface_inf(test_img, model)
cv2.imwrite('./result.jpg', test_img)