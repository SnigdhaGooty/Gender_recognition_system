# Gender Recognition (Face-based)

Project that trains a small CNN to classify face images by gender and runs a real-time demo using YOLOv8 for face detection and a Keras model for gender prediction.

## Repository contents
- [Gender_recog.ipynb](Gender_recog.ipynb) — Jupyter notebook used to prepare data and train the CNN; downloads/extracts a zip archive, creates ImageDataGenerators, defines/trains the model, and saves `gender_model.keras`.
- [real_time.py](real_time.py) — Real-time demo: captures webcam frames, runs YOLOv8 face detection, crops faces, resizes to 128×128, normalizes pixels, predicts gender with the saved Keras model, and overlays labels.
- gender_model.keras — Saved Keras model produced by the notebook (binary sigmoid output: >0.5 => Male).
- yolov8n-face-lindevs.pt — YOLOv8 face detection weights used by `ultralytics.YOLO`.

## How it works (end-to-end)
1. Data: the notebook expects a training and validation folder structure under Training/ and Validation/ where class subfolders (e.g., Male, Female) contain face images.
2. Training: `Gender_recog.ipynb` uses `ImageDataGenerator(rescale=1./255)` and `flow_from_directory` with target size 128×128 and `class_mode='binary'`.
3. Model: a Sequential CNN with three Conv2D+MaxPool blocks, a Flatten, Dense(64, relu), and Dense(1, sigmoid). Compiled with `adam` and `binary_crossentropy` and trained for 15 epochs (as in the notebook).
4. Save: model saved as `gender_model.keras` using `model.save(...)`.
5. Inference: `real_time.py` loads YOLO face detector and the saved Keras model; for each detected face it crops, resizes to 128×128, scales pixel values (divide by 255), expands dims, then runs `model.predict` to get a probability; threshold 0.5 maps to Male/Female.

## Dependencies
- Python 3.8+
- tensorflow (tested with 2.10+)
- ultralytics (YOLOv8 python package)
- opencv-python
- numpy

Install with:

```bash
python -m pip install tensorflow ultralytics opencv-python numpy
```

## Running the project

1. Training (Notebook): Open [Gender_recog.ipynb](Gender_recog.ipynb) in Colab or Jupyter and run cells in order. Ensure your Training/ and Validation/ folders are available or upload an archive as the notebook expects.

2. Real-time demo (local):

```bash
# ensure dependencies installed
python real_time.py
```

Make sure `gender_model.keras` and `yolov8n-face-lindevs.pt` are in the same folder or update paths inside `real_time.py`.

## Notes, limitations and improvements
- Dataset quality & bias: model outcome depends heavily on training data. Balanced, diverse data is required to reduce bias.
- Face detector: YOLOv8 provides fast face boxes; accuracy of gender prediction depends on clean, frontal crops.
- Small model: the CNN used is intentionally small — good for quick prototyping but may underperform on real-world variability.
- Improvements: data augmentation, transfer learning (e.g., MobileNetV2 or EfficientNet), multi-class outputs, evaluation metrics (precision/recall, confusion matrix), calibration, and privacy-aware deployment.

