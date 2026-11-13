
import os
import math
import torch
import numpy as np
from PIL import Image
from pathlib import Path

import serial as ser
import time

# from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import ( check_img_size, cv2,non_max_suppression,  scale_coords,set_logging )
from models.experimental import attempt_load
from utils.torch_utils import select_device,time_sync
from utils.augmentations import letterbox
from utils.plots import Annotator,colors

def run_webcam(save_path=None, img_size=640, stride=32, augment=False, visualize=False):

    # serial with arduino
    se = ser.Serial("/dev/ttyS0",115200,timeout=1)

    device = 0

    # people
    # weights = r'yolov5n.engine'
    # data='data/coco128.yaml'

    light_angel_bias = 0

    imgsz=(640, 640)
    
    # car 
    weights = r'best_SGD_2612.engine'
    data='coco128_my.yaml'


    # # Initialize
    # set_logging()
    # device = select_device(0)
    # import .pt model 
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    img_size = check_img_size(img_size, s=stride)
    names = model.names
    half = True
    # Half
    half &= (pt or jit or onnx or engine) and 1  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    cap = cv2.VideoCapture(1,cv2.CAP_V4L2)
    cap1 = cv2.VideoCapture(2,cv2.CAP_V4L2)
    width = 640
    height = 480
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # print(cap.get(cv2.CAP_PROP_FPS) % 100)
    frame = 0
    fps = 0

    ret_val, img0 = cap.read()
    ret_val1, img1 = cap1.read()
    # fps, w, h = 30, img0.shape[1], img0.shape[0]
    save_path = 'output.mp4'
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width*2, height))
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz), half=True)  # warmup
    aaa = []
    ttt=0
    aaaa=0
    while True:
        ret_val, img0 = cap.read()
        ret_val1, img11 = cap1.read()
        img0 = cv2.flip(img0,-1)
        img11 = cv2.flip(img11,-1)

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cap1.release()
            cv2.destroyAllWindows()
            break

        if not ret_val or not ret_val1:
            break

        frame += 1
        # print(f'video {frame} {save_path}')
        t1 = time_sync()
        img = img0.copy()
        img1 = img11.copy()
        stride=32
        # Padded resize
        img = letterbox(img0, 640, stride=stride, auto=False)[0]
        img1 = letterbox(img11, 640, stride=stride, auto=False)[0]
        # img = [letterbox(x, 640, stride=stride, auto=False)[0] for x in img0]
        # # Stack
        # img = np.stack(img, 0)
        # # Convert
        # img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        # img = np.ascontiguousarray(img)


        # img1 = [letterbox(x, 640, stride=stride, auto=False)[0] for x in img11]
        # # Stack
        # img1 = np.stack(img1, 0)
        # # Convert
        # img1 = img1[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        # img1 = np.ascontiguousarray(img1)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)

        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img1 = np.ascontiguousarray(img1)

        img = torch.from_numpy(img.copy()).to(device)
        img = img.half() / 255.0   # 0 - 255 to 0.0 - 1.0
        img = img[None]     # [h w c] -> [1 h w c]

        img1 = torch.from_numpy(img1.copy()).to(device)
        img1 = img1.half() / 255.0   # 0 - 255 to 0.0 - 1.0
        img1 = img1[None]     # [h w c] -> [1 h w c]
        

        
        # inference
        pred = model(img, augment=augment)
        # pred = model(img, augment=augment, visualize=visualize)[0]
        pred = non_max_suppression(pred, conf_thres=0.2, iou_thres=0.55)

        pred1 = model(img1, augment=augment)
        # pred1 = model(img1, augment=augment, visualize=visualize)[0]
        pred1 = non_max_suppression(pred1, conf_thres=0.2, iou_thres=0.55)
        
        
        # plot label webcam0
        det = pred[0]
        annotator = Annotator(img0.copy(), line_width=3, example=str(names))
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'

                # 單獨class
                # if c!=0:
                #     continue
                
                # print("左上xy:",xyxy[0].cpu().numpy(),xyxy[1].cpu().numpy(),"右下xy:",xyxy[2].cpu().numpy(),xyxy[3].cpu().numpy()," class:",label)
                annotator.box_label(xyxy, label, color=colors(c, True))

        # plot label webcam1
        det1 = pred1[0]
        annotator1 = Annotator(img11.copy(), line_width=3, example=str(names))
        if len(det1):
            det1[:, :4] = scale_coords(img1.shape[2:], det1[:, :4], img11.shape).round()
            for *xyxy, conf, cls in reversed(det1):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'

                # 單獨class
                # if c!=0:
                #     continue

                
                # y-axis avg
                # if(c==45):
                #     if len(aaa)==100:
                #         break
                #     else:
                #         aaa.append(xyxy[1].cpu().numpy()-xyxy[3].cpu().numpy()) 

                # print("左上xy:",xyxy[0].cpu().numpy(),xyxy[1].cpu().numpy(),"右下xy:",xyxy[2].cpu().numpy(),xyxy[3].cpu().numpy()," class:",label)
                annotator1.box_label(xyxy, label, color=colors(c, True))


        # read image
        im0 = annotator.result()
        im1 = annotator1.result()
        image_merge = np.concatenate((im0,im1),axis=1)
        # image_merge = np.concatenate((img0,img11),axis=1)
        
        angle_list = []
        time_start = time.time()
        same_object_argment = same_object_detect(det,det1,names,img0,img11)
        # print(f"{time.time()-time_start}")
        for pred1_xy, pred2_xy, c_number,same_number_id_add, object_distance,angle in same_object_argment:
            # cv2.putText(img, c_number+str(same_number_id_add)+" dis:"+str(int(object_distance))+"cm", (int(pred1_xy[0]), int((pred1_xy[1]+pred1_xy[3])//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            angle_list.append(angle)
            cv2.putText(image_merge, (c_number+str(same_number_id_add))[2]+" dis:"+str(round(object_distance,2))+"m", (int(pred2_xy[0]+640), int((pred2_xy[1]+pred2_xy[3])//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if angle_list:
            aaaa+=1
            if(ttt>0):
                aaaa=0
            if(aaaa>5):
            # print(str(-(90-round(min(angle_list)))))
                se.write(str(-(round(90-min(angle_list),2))+light_angel_bias).encode())
            # print((round(90-min(angle_list),2)))
            ttt=0
        else:
            ttt+=1
            if(ttt>9):
                se.write(str(0).encode())
        
        t2 = time_sync()       

        
        # # put text on same target
        # for pred1_xy, pred2_xy, c_number in same_object_argment:
        #  qrint(((pred1_xy[0]+pred1_xy[2])//2, (pred1_xy[1]+pred1_xy[3])//2),c_number)
        #     # print(((pred2_xy[0]+pred2_xy[2])//2, (pred2_xy[1]+pred2_xy[3])//2),c_number)
        #     cv2.putText(image_merge, c_number, (int((pred1_xy[0]+pred1_xy[2])//2), int((pred1_xy[1]+pred1_xy[3])//2)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 3, cv2.LINE_AA)
        #     cv2.putText(image_merge, c_number, (int((pred2_xy[0]+pred2_xy[2])//2)+640, int((pred2_xy[1]+pred2_xy[3])//2)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 3, cv2.LINE_AA)
        fps += (t2 - t1)
        # print(f'Done. ({t2 - t1:.3f}s)')
        cv2.putText(image_merge, str(int(1/(fps/frame))), (10,100), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 3, cv2.LINE_AA)
        vid_writer.write(image_merge)
        cv2.imshow('webcam', image_merge)
        cv2.waitKey(1)
    # print(f"{max(aaa)} {min(aaa)}")
    # print((max(aaa)-min(aaa))/(sum(aaa)/100))
    vid_writer.release()
    cap.release()
    cap1.release()
    cv2.destroyAllWindows()


def pHash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))   # , interpolation=cv2.INTER_CUBIC
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]
 
    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

# 圖像相似度
def runImageSimilaryFun(para1, para2):
    hash1 = pHash(para1)
    hash2 = pHash(para2)
    n3 = cmpHash(hash1, hash2)
    # print('感知哈希算法相似度pHash：', n3)
    return 1-float(n3/64)
    print('準確: ',1-float(n3/64))


# 相同物體位置匹配
def same_object_detect(pred1, pred2,names, webcam0_image:np.ndarray,webcam1_image:np.ndarray):
    # 所有物體的class number
    pred1_all_items_class = []
    pred2_all_items_class = []
    # 所有物體的xy
    pred1_all_items_xy = []
    pred2_all_items_xy = []

    # 相同物體的在pred1,2中的index和class
    same_object_argment = []

    for *xyxy, conf, cls in reversed(pred1):
        c = int(cls)  # integer class
        pred1_all_items_xy.append([xyxy[0].cpu().numpy(),xyxy[1].cpu().numpy(),xyxy[2].cpu().numpy(),xyxy[3].cpu().numpy()])
        pred1_all_items_class.append(c)
        # label = f'{names[c]} {conf:.2f}'

    for *xyxy, conf, cls in reversed(pred2):
        c = int(cls)  # integer class
        pred2_all_items_xy.append([xyxy[0].cpu().numpy(),xyxy[1].cpu().numpy(),xyxy[2].cpu().numpy(),xyxy[3].cpu().numpy()])
        pred2_all_items_class.append(c)
        # label = f'{names[c]} {conf:.2f}'

    item_index0 = np.array(pred1_all_items_class)
    item_index1 = np.array(pred2_all_items_class)

    # 比較兩者相同的class
    same_item_class = set(pred1_all_items_class) & set(pred2_all_items_class)

    # 所有具有相同class的進行一次比較
    for class_num in same_item_class:
        # 搜尋相同的class number
        if class_num!=0:
            continue
        index0 = np.where(item_index0==class_num)[0]
        index1 = np.where(item_index1==class_num)[0]
        same_number_id_add=0
        # 比較兩者xy與矩形大小
        for pred1_index  in index0:
            mmse = []
            mmse_index = []
            same_number_id_add += 1
            # 兩台相機距離 (cm)
            base = 30
            # object distance (m)
            z_distance = 0
            # unit (m)
            height_from_ground = 0.86

            pred1_index= int(pred1_index)
            pred1_left_x = int(pred1_all_items_xy[pred1_index][0])
            pred1_left_y = int(pred1_all_items_xy[pred1_index][1])
            pred1_right_x = int(pred1_all_items_xy[pred1_index][2])
            pred1_right_y = int(pred1_all_items_xy[pred1_index][3])
           
            for pred2_index in index1:
                pred2_index = int(pred2_index)

                pred2_left_x = int(pred2_all_items_xy[pred2_index][0])
                pred2_left_y = int(pred2_all_items_xy[pred2_index][1])
                pred2_right_x = int(pred2_all_items_xy[pred2_index][2])
                pred2_right_y = int(pred2_all_items_xy[pred2_index][3])

                # 相似度計算
                # print("相似度: ",runImageSimilaryFun(webcam0_image[pred1_left_y:pred1_right_y,pred1_left_x:pred1_right_x,:], webcam1_image[pred2_left_y:pred2_right_y,pred2_left_x:pred2_right_x, :]))

                
                # 左上的y1和y2相同 右下的y1和y2相同 在同一條水平線上
                obj1_mid_x = (pred1_left_x+pred1_right_x)/2
                obj2_mid_x = (pred2_left_x+pred2_right_x)/2

                # 物體一和物體二的x軸長度
                x1_length = abs(pred1_left_x-pred1_right_x)
                x2_length = abs(pred2_left_x-pred2_right_x)
                # 物體一和物體二的y軸長度
                y1_length = abs(pred1_left_y-pred1_right_y)
                y2_length = abs(pred2_left_y-pred2_right_y)

                # 畫面邊緣物體不列入計算
                if pred1_left_x<10 or pred1_right_x>630 or pred2_left_x<10 or pred2_right_x>630:
                    continue
                # 左畫面物體比右畫面物體x軸更大
                if not obj1_mid_x>obj2_mid_x:
                    continue
                # 兩個畫面水平位置誤差不超過畫面高度的5%
                if not (abs((pred1_left_y+pred1_right_y)/2-(pred2_left_y+pred2_right_y)/2))<=24:
                    continue
                # 相似度計算
                zzz = runImageSimilaryFun(webcam0_image[pred1_left_y:pred1_right_y,pred1_left_x:pred1_right_x,:], webcam1_image[pred2_left_y:pred2_right_y,pred2_left_x:pred2_right_x, :])
                # print(zzz)
                if zzz<0.7:
                    continue
                # if abs(x1_length-x2_length)<=50:
                #     mmse.append(mmse_lgorithm(x1_length,x2_length,y1_length,y2_length))
                #     z_distance =  estimation_distance2(obj1_mid_x,obj2_mid_x,base)
                #     print(f"--{names[class_num]} {same_number_id_add}-----",z_distance,"m")
                #     angle = math.degrees(math.atan(z_distance/height_from_ground))
                #     mmse_index.append([pred1_index,pred2_index,z_distance,angle])

                
                z_distance =  estimation_distance2(obj1_mid_x,obj2_mid_x,base)
                print(f"--{names[class_num]} {same_number_id_add}-----",z_distance,"m")
                angle = math.degrees(math.atan(z_distance/height_from_ground))
                if z_distance == 0 :
                    continue
                
                mmse.append(mmse_lgorithm(x1_length,x2_length,y1_length,y2_length))
                mmse_index.append([pred1_index,pred2_index,z_distance,angle])
                # print(mmse)
                    # print(f"{names[class_num]} {same_number_id_add} angle:----",angle," degree")
                    # same_object_argment.append([pred1_all_items_xy[pred1_index], pred2_all_items_xy[pred2_index], names[class_num],same_number_id_add,z_distance,angle])
            if mmse:
                a = mmse.index(min(mmse))
                same_object_argment.append([pred1_all_items_xy[mmse_index[a][0]], pred2_all_items_xy[mmse_index[a][1]], names[class_num],same_number_id_add,mmse_index[a][2],mmse_index[a][3]])

    return same_object_argment


def estimation_distance2(object1_x,object2_x,b):
    x0 = 640
    if(object1_x-object2_x)==0:
        return 0
    distance = b*x0/(math.tan(math.radians(30))*2*(object1_x-object2_x))
    return distance/100.0

def mmse_lgorithm(wl,wr,h1,hr):
    p=0.5
    return p*((wl-wr)**2) + (1-p)*((h1-hr)**2)




if __name__ == '__main__':
    run_webcam()


