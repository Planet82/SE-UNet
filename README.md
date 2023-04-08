# SE-UNet
### 所需环境
torch==1.2.0    
torchvision==0.4.0   
### 训练步骤

#### 训练自己的数据集
1、在训练前利用voc_annotation.py文件生成对应的txt。    
2、注意修改train.py的num_classes为分类个数+1。 
3、提前下载好预训练权重
4、运行train.py即可开始训练。  

### 预测步骤
#### 使用自己训练的权重
1. 按照训练步骤训练。    
2. 在unet.py文件里面，在如下部分修改model_path、backbone和num_classes使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
3. 运行predict.py，输入    
```python
Test_Images/t_11.jpg
```   
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。    

### 得到评价指标
1、设置get_miou.py里面的num_classes为预测的类的数量加1。  
2、设置get_miou.py里面的name_classes为需要去区分的类别。  
3、运行get_miou.py即可获得miou大小。  

###由于部分文件过大，我将上传到百度网盘中
预训练权重
链接：https://pan.baidu.com/s/1m_JJFL0IbxzISwN81Ve3Eg 
提取码：xdlz
