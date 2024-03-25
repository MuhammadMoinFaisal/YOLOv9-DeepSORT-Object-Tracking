<H1 align="center">
YOLOv9 Object Detection with DeepSORT Tracking(ID + Trails) </H1>

### New Features
- Added Label for Every Track
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

## Ready to Use Google Colab
[`Google Colab File`](https://colab.research.google.com/drive/1IivrmAtnhpQ1PSmWsp-G6EnqsUOol9v8?usp=sharing)

## Steps to run Code

- Clone the repository
```
git clone https://github.com/MuhammadMoinFaisal/YOLOv9-DeepSORT-Object-Tracking.git
```
- Goto the cloned folder.
```
cd YOLOv9-DeepSORT-Object-Tracking
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Download the pre-trained YOLOv9 model weights
[yolov9](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt)

- Downloading the DeepSORT Files From The Google Drive 
```
gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
```
- After downloading the DeepSORT Zip file from the drive, unzip it. 

- Download sample videos from the Google Drive
```
gdown "https://drive.google.com/uc?id=115RBSjNQ_1zjvKFRsQK2zE8v8BIRrpdy&confirm=t"
gdown "https://drive.google.com/uc?id=1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5&confirm=t"
```
```
# for detection only
python detect_dual.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0

#for detection and tracking
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0

#for WebCam
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 0 --device 0

#for External Camera
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 1 --device 0

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source "your IP Camera Stream URL" --device 0

#for specific class (person)
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --classes 0

#for detection and tracking with trails 
!python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --draw-trails 
```

- Output file will be created in the ```working-dir/runs/detect/obj-tracking``` with original filename


### Watch the Complete Step by Step Explanation

- Video Tutorial Link  [`YouTube Link`](https://www.youtube.com/watch?v=Jx6oLBfDxRo)


[![Watch the Complete Tutorial for the Step by Step Explanation](https://img.youtube.com/vi/Jx6oLBfDxRo/0.jpg)]([https://www.youtube.com/watch?v=Jx6oLBfDxRo](https://www.youtube.com/watch?v=Jx6oLBfDxRo))

