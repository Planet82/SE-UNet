import os
from PIL import Image
from tqdm import tqdm
import numpy as np

from unet import Unet
from utils.medical_metrics import compute_accuracy, compute_iou, compute_recall_precision

if __name__ == "__main__":
    
    images_path = 'Test_Images'
    labels_path = 'Test_Labels'
    
    images          = os.listdir(images_path)
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
        
    print("Load model.")
    unet = Unet()
    print("Load model done.")

    print("Get predict result.")
    for image in tqdm(images):
        image_path  = os.path.join(images_path, image)
        inputimage      = Image.open(image_path)
        outputimage       = unet.detect_image(inputimage)
        arr = np.array(outputimage)
        # 对每个像素的RGB值进行平均
        arr_avg = np.round(np.mean(arr, axis=2)).astype(np.uint8)
        # 将结果转换为8位整数类型
        outputimage = Image.fromarray(arr_avg)
        outputimage.save(os.path.join(pred_dir, image[:-4]+".png"))

    print("Get predict result done.")

    # 定义标签和分割结果文件路径
    results_path = 'miou_out/detection-results'
    labels = os.listdir(labels_path)

    mIoU = 0
    mAccuracy = 0
    mPrecision = 0
    mRecall = 0
    for label in labels:
        # 加载标签和分割结果
        gt = Image.open(os.path.join(labels_path, label))
        pred = Image.open(os.path.join(results_path, label))

        gt = np.array(gt)
        pred = np.array(pred)
        
        # 计算mIoU和Accuracy
        iou = compute_iou(pred, gt)
        acc = compute_accuracy(pred, gt)
        
        # 计算Recall和Precision
        recall,precision = compute_recall_precision(pred, gt)

        # 输出结果
        # print(label,"||" ,'IoU:', iou ,"||", 'Accuracy:', acc,"||",'Recall:', recall,"||",'Precision:', precision)
        mIoU += iou 
        mAccuracy += acc
        mPrecision += precision
        mRecall += recall
    num = len(labels)
    print("mIOU:",mIoU/num, "||" , "mAccuracy:",mAccuracy/num, "||" , "mPrecision:",mPrecision/num, "||" , "mRecall:",mRecall/num)
    with open("result.txt","w") as f:
        f.writelines("mIOU:"+str(mIoU/num)+"  ||  "+"mAccuracy:"+str(mAccuracy/num)+"  ||  " +"mPrecision:"+str(mPrecision/num)+ "  ||  " + "mRecall:"+str(mRecall/num))
        f.close()

