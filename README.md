# Autonomous Driving Application - Car Detection

A deep learning project for object detection in autonomous driving scenarios using YOLO (You Only Look Once) v2 model. This project implements car detection and general object detection capabilities using a pre-trained YOLO model converted from Darknet to Keras format.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Exercises and Learning Objectives](#exercises-and-learning-objectives)
- [Dataset](#dataset)
- [References](#references)
- [License](#license)

## 🎯 Overview

This project is designed for learning and implementing object detection in the context of autonomous driving. It uses the YOLO v2 algorithm, which achieves high accuracy while running in real-time. The model can detect 80 different object classes from the COCO dataset, with a primary focus on car detection for self-driving car applications.

The project includes:
- A complete Jupyter notebook with guided exercises
- Pre-trained YOLO v2 model weights
- Utilities for image and video processing
- Implementation of key YOLO components (non-max suppression, IoU, bounding box filtering)

## ✨ Features

- **Real-time Object Detection**: Detect objects in images and videos using YOLO v2
- **80 Object Classes**: Support for 80 different object classes from COCO dataset
- **Car Detection**: Specialized focus on detecting cars in driving scenarios
- **Video Processing**: Process video files frame by frame for object detection
- **Bounding Box Visualization**: Draw bounding boxes and labels on detected objects
- **Non-Max Suppression**: Implement advanced filtering to improve detection accuracy
- **Intersection over Union (IoU)**: Calculate overlap between bounding boxes

## 📦 Requirements

### Software Requirements

- Python 3.6+
- Jupyter Notebook
- TensorFlow 1.x or 2.x
- Keras 2.0+
- NumPy
- OpenCV (cv2)
- Pillow (PIL)
- Matplotlib
- Pandas
- h5py

### Hardware Recommendations

- GPU support recommended for faster inference (TensorFlow-GPU)
- Minimum 4GB RAM
- Sufficient storage for model weights (~250MB)

## 🚀 Installation

### Option 1: Using Conda Environment (Recommended)

```bash
# Navigate to the yad2k directory
cd yad2k

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate yad2k
```

### Option 2: Manual Installation

```bash
# Install required packages
pip install numpy h5py pillow matplotlib opencv-python pandas

# Install TensorFlow (CPU version)
pip install tensorflow

# Or install TensorFlow with GPU support
pip install tensorflow-gpu

# Install Keras
pip install keras
```

### Verify Installation

```bash
python -c "import tensorflow as tf; import keras; print('Installation successful!')"
```

## 📁 Project Structure

```
Autonomous_Driving_Application_Car_Detection/
│
├── Autonomous_driving_application_Car_detection.ipynb  # Main Jupyter notebook
├── yolov2.h5                                           # Pre-trained YOLO v2 model weights
├── coco.names                                          # COCO dataset class names
├── Instruction.txt                                     # Assignment instructions
│
├── images/                                            # Test images directory
│   └── test.jpg
│
├── videos/                                            # Test videos directory
│   └── Test_video.mp4
│
├── out/                                               # Output directory for results
│   ├── test.jpg
│   └── Test_video_detected.mp4
│
├── model_data/                                        # Model configuration files
│   ├── coco_classes.txt                               # COCO class labels
│   ├── pascal_classes.txt                             # Pascal VOC class labels
│   ├── yolo_anchors.txt                               # YOLO anchor box definitions
│   └── _anchors.txt
│
├── nb_images/                                         # Notebook images and diagrams
│   ├── architecture.png
│   ├── anchor_map.png
│   ├── box_label.png
│   ├── iou.png
│   ├── non-max-suppression.png
│   └── ...
│
├── font/                                              # Font files for visualization
│   └── FiraMono-Medium.otf
│
└── yad2k/                                             # YAD2K library (Darknet to Keras converter)
    ├── README.md
    ├── environment.yml
    ├── yad2k.py                                       # Model converter script
    ├── test_yolo.py                                   # Testing script
    ├── retrain_yolo.py                                # Retraining script
    ├── yad2k/                                         # Main package
    │   ├── models/
    │   │   ├── keras_yolo.py                          # YOLO model implementation
    │   │   └── keras_darknet19.py                     # Darknet-19 backbone
    │   └── utils/
    │       ├── draw_boxes.py                          # Visualization utilities
    │       └── utils.py                               # Helper functions
    └── voc_conversion_scripts/                        # Dataset conversion scripts
```

## 💻 Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**:
   - Open `Autonomous_driving_application_Car_detection.ipynb`

3. **Follow the exercises**:
   - The notebook contains step-by-step exercises to implement YOLO components
   - Complete the graded functions marked with `#GRADED FUNCTION`
   - Run cells sequentially to build and test your implementation

### Key Exercises

The notebook includes the following main exercises:

1. **`yolo_filter_boxes`**: Filter boxes based on class scores threshold
2. **`iou`**: Calculate Intersection over Union between two boxes
3. **`yolo_non_max_suppression`**: Implement non-max suppression to remove duplicate detections
4. **`yolo_eval`**: Complete YOLO evaluation pipeline

### Testing on Images

```python
# Load the pre-trained model
from tensorflow.keras.models import load_model
model = load_model('yolov2.h5')

# Use the utilities from yad2k
from yad2k.yad2k.models.keras_yolo import yolo_head
from yad2k.yad2k.utils.draw_boxes import draw_boxes
from yad2k.yad2k.utils.utils import preprocess_image, scale_boxes, read_classes, read_anchors

# Process an image (see notebook for complete example)
```

### Testing on Videos

The notebook includes code to process video files frame by frame and save the output with bounding boxes drawn on detected objects.

### Using the Command Line Tool

You can also use the provided test script:

```bash
cd yad2k
python test_yolo.py ../yolov2.h5 \
    --anchors_path ../model_data/yolo_anchors.txt \
    --classes_path ../model_data/coco_classes.txt \
    --test_path ../images \
    --output_path ../out
```

## 🔬 Model Details

### YOLO v2 Architecture

- **Input**: Images of shape (608, 608, 3)
- **Backbone**: Darknet-19
- **Output**: Bounding boxes with class predictions
- **Grid Size**: 19×19
- **Anchor Boxes**: 5 anchor boxes per grid cell
- **Classes**: 80 COCO classes

### Encoding Format

Each bounding box is represented by 85 numbers:
- `p_c`: Probability that there's an object
- `b_x, b_y, b_h, b_w`: Bounding box coordinates (center x, center y, height, width)
- `c`: 80-dimensional class probability vector

### Anchor Boxes

The model uses 5 pre-defined anchor boxes stored in `model_data/yolo_anchors.txt`:
```
0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
```

## 📚 Exercises and Learning Objectives

By completing this project, you will learn to:

- ✅ Understand the YOLO object detection algorithm
- ✅ Implement non-max suppression to improve detection accuracy
- ✅ Calculate Intersection over Union (IoU) for bounding boxes
- ✅ Filter detections based on confidence scores
- ✅ Handle bounding box coordinates and transformations
- ✅ Apply object detection to real-world images and videos
- ✅ Work with pre-trained deep learning models

## 🗂️ Dataset

### COCO Dataset Classes

The model is trained on the COCO (Common Objects in Context) dataset and can detect 80 object classes including:
- Vehicles: car, bus, truck, motorbike, bicycle, train, boat, aeroplane
- People: person
- Animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- Common objects: chair, sofa, bed, diningtable, toilet, tvmonitor, laptop, mouse, keyboard, cell phone, book, clock, vase, scissors, and many more

See `model_data/coco_classes.txt` or `coco.names` for the complete list.

### Test Data

- Sample images in `images/` directory
- Sample video in `videos/` directory
- Dataset provided by [drive.ai](https://www.drive.ai/)

## 📖 References

### Papers

1. **YOLO v1**: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

2. **YOLO v2**: Redmon, J., & Farhadi, A. (2016). [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

### Resources

- [YAD2K Repository](https://github.com/allanzelener/yad2k) - Yet Another Darknet 2 Keras
- [Darknet](https://github.com/pjreddie/darknet) - Original YOLO implementation
- [COCO Dataset](https://cocodataset.org/) - Common Objects in Context
- [Drive.ai](https://www.drive.ai/) - Autonomous driving dataset provider

## 📝 Important Notes

### For Assignment Submission

Before submitting your assignment, ensure:

1. ❌ No extra `print` statements added
2. ❌ No extra code cells added
3. ❌ Function parameters are not modified
4. ❌ No global variables used (unless explicitly instructed)
5. ❌ No unnecessary code changes

### Code Style Guidelines

- Avoid using loops (for/while) unless explicitly required
- Follow the structure provided in the notebook
- Complete graded functions between `### START SOLUTION HERE ###` and `###END SOLUTION HERE###` markers
- Test your code after each exercise to ensure correctness

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure you're in the correct directory
   - Check that all dependencies are installed
   - Verify Python path includes the project root

2. **Model Loading Errors**:
   - Verify `yolov2.h5` exists in the root directory
   - Check file permissions

3. **Memory Issues**:
   - Reduce image resolution if running out of memory
   - Process videos in smaller chunks
   - Use CPU version if GPU memory is limited

4. **Video Processing Errors**:
   - Ensure OpenCV is properly installed
   - Check video codec compatibility
   - Verify output directory exists and is writable

## 📄 License

This project includes multiple licenses:
- See `LICENSE` file in the root directory
- See `Drive.ai Dataset Sample LICENSE` for dataset licensing
- See `yad2k/LICENSE` for YAD2K library license
- Font licenses in `font/SIL Open Font License.txt`

## 🤝 Contributing

This is an educational project. For improvements or bug fixes:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 👨‍💻 Author

This project is part of a course on Convolutional Neural Networks and Deep Learning.

---

**Happy Learning! 🚗🤖**

For questions or issues, please refer to the course materials or create an issue in the repository.
