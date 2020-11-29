# 基于 Greengrass 端侧实时人脸检测
Greengrass很容易部署在设备侧/网关侧，同时也提供良好的运行时环境，针对安防监控厂商Camera设备可以结合Greengrass来实现边缘侧AI/ML场景。这里通过树莓派部署Greengrass跑dlib库从摄像机实时视频流中抽取视频帧来实现人脸识别和比对。

## 准备工作：
- 一台树莓派设备，本方案采用RaspberryPi 4B ,CSI摄像头。
- 将CSI摄像头处理为Raspbian OS能识别的设备，需开启V4l2 Module
- 树莓派上安装python3运行环境
- 安装Greengrass，参考官方文档，这里创建名称为“raspberrypiGroup”组
- 上传一张照片到树莓派指定目录下用于后续人脸比对


