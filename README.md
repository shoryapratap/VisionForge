# 🛰 Vision Forge

### 🚀 Real-Time Object Detection for Space Stations (YOLOv8 + TFLite)

A powerful object detection system designed for space station environments, built during the *HackByte – Space Station Hackathon*.  
Vision Forge leverages synthetic simulation data and real-time detection powered by YOLOv8, and is optimized for mobile deployment via Flutter using TensorFlow Lite.

📥 [Download Dataset](https://your-download-link.com/dataset.zip)

---

## 🧠 Project Summary

- *Model*: YOLOv8 Small (yolov8s.pt)
- *Export Format*: TFLite (best_float32.tflite)
- *Interface*: Flutter Mobile App
- *Dataset*: Synthetic space station images generated via Falcon simulation platform
- *Purpose*: Detect tools, devices, and components in a zero-gravity environment

---

## ⚙ How to Use

### 🐍 Python AI Project

bash
# Clone Repo
git clone https://github.com/yourusername/vision-forge.git
cd DualityAI_Hackathon_Submission/1_Python_AI_Project/

# (Optional) Set up Python env
python -m venv env
source env/bin/activate

# Install Dependencies
pip install -r requirements.txt
`

### 🏋 Train the Model

bash
python train.py \
  --epochs 5 \
  --mosaic 0.1 \
  --optimizer AdamW \
  --momentum 0.2 \
  --lr0 0.001 \
  --lrf 0.0001 \
  --single_cls False


Weights are saved to:  
runs/detect/Final_Run_78mAP/weights/best.pt

---

### 🔎 Predict + Evaluate

bash
python predict.py


- Annotated results are saved to predictions/images/
    
- YOLO-format labels are saved to predictions/labels/
    
- Automatic evaluation: Precision, Recall, mAP@0.5, Confusion Matrix
    

---

### 👁 Visualize Dataset

bash
python visualize.py


Controls:

- a / d → Previous / Next image
    
- t / v → Train / Val set toggle
    
- q → Quit viewer
    

---

## 📈 Results (Before vs After)

|Metric|Default Code|Vision Forge|
|---|---|---|
|mAP@0.5|0.61|*0.78* ✅|
|Precision|0.63|*0.80* ✅|
|Recall|0.59|*0.75* ✅|
|F1-Score|0.60|*0.77* ✅|

---

## 🛠 Challenges & Solutions

|Problem|Solution|
|---|---|
|High overfitting|Reduced mosaic augmentation to 0.1|
|Label mismatches|Built visualizer to cross-verify annotations|
|Complex configs|Used yolo_params.yaml for clarity|
|Ambiguous model evaluation|Added automated metrics via predict.py|

---

## 📱 Mobile Integration (Flutter)

- Export your trained model:
    
    bash
    yolo export model=best.pt format=tflite
    
    
- Place best_float32.tflite into Flutter/assets/
    
- Include labels.txt beside the model
    
- Load using TFLite plugin in main.dart for real-time inference
    

---

## 💡 Future Enhancements

- Real-time camera inference on device
    
- Web dashboard for monitoring predictions
    
- Multi-class support and object tracking
    
- Deploy to embedded edge devices
    

---

## 🙌 Team HackInHub

Built with collaboration, innovation, and a deep passion for AI in extreme environments for *HackByte – Space Station Hackathon*.

*Team Name*: HackInHub 🛠  
*Project*: Vision Forge

- 👨‍🚀 Shorya Pratap
    
- 🧠 Yogita Chugh
    
- 🔧 Krishna Gupta
    
- 💡 Lakshya Goyal
