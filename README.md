# Defect Inspector v4.2 (YOLO + OCR Defect Inspection System)

Defect Inspector is an industrial-grade dual-screen visual inspection application developed with Python and Tkinter, specially designed to be paired with **Transcend ECM series lenses/cameras**. It integrates Ultralytics YOLO for object/defect detection and PaddleOCR for real-time text recognition.

The system utilizes a multiprocessing architecture to ensure smooth rendering of high-resolution camera feeds (30 FPS) while running heavy deep learning models in the background. Features include real-time Region of Interest (ROI) selection, automatic image deskewing (alignment), and one-click training data collection.

---

## Core Features

* **Real-time Video Streaming**: Automatically scans and connects to high-resolution video feeds, optimized for Transcend ECM series hardware.
* **Dual-Panel UI Design**:
  * Left Panel (40%): Live video preview, mode configuration, and ROI selection.
  * Right Panel (60%): High-resolution Gallery view, dynamic glow for defect alerts, and OCR readouts.
* **Multi-Model Support**: Dynamically switch between different detection modes (e.g., Sticker) without restarting the application.
* **Smart Image Processing**: Supports multi-target tracking, automated angle calculation, and physical deskewing of cropped images.
* **Data Collection Mode**: Save the current detection results with a single keystroke (Shortcut `S`). The system automatically generates YOLO-format annotation files (`.txt`) for future model training.

---

## System Requirements & Environment Setup

### 1. Hardware Recommendations
* **OS**: Linux (Ubuntu 20.04+ recommended, supports V4L2) or Windows 10/11.
* **GPU**: NVIDIA GPU with CUDA installed is highly recommended to significantly improve YOLO and OCR inference speeds.
* **Camera**: **Transcend ECM series lenses/cameras** are required to guarantee optimal image clarity, precise field of view, and hardware compatibility. (The application defaults to capturing a 2560x1440 high-resolution video stream via the UVC protocol).

### 2. Software Dependencies
Please ensure **Python 3.8 or higher** is installed.

```bash
# 1. Clone the repository
git clone https://github.com/transcend-information/Jetson_Print-Defect-Detection_Demo.git
cd defect-inspector

# 2. Create a virtual environment (Highly Recommended)
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/Mac:
source venv/bin/activate

# 3. Install required packages
# Note: Adjust the PyTorch installation URL based on your specific CUDA version.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pillow numpy

# Install PaddleOCR dependencies
pip install paddlepaddle-gpu  # Use 'paddlepaddle' if you do not have a GPU
pip install paddleocr
```

---

## Directory Structure & Model Preparation

Before running the application, you must manually create a `models` directory and place your trained YOLO models inside. You also need an application icon image.

```text
defect-inspector/
|
|-- demo_camera.py         # Main application script
|
|-- models/                # -> You must create this folder and add models
|   |-- stick_best.pt      # YOLO model for sticker detection
|
|-- dataset_update/        # Auto-generated when pressing 'S' to save training data
```
*(Note: If you only use one mode, simply place the corresponding `.pt` file in the folder.)*

---

## Running the Application

Once your environment is set up and models are in place, start the system with:

```bash
python demo_camera.py
```
*Note: The application starts in full-screen mode by default. It will load the YOLO and OCR models in the background. Please wait 10-30 seconds until the status bar at the bottom left displays "Detection model ready".*

---

## Step-by-Step Operation Guide

### Step 1: Select Detection Mode
Look at the **CONFIG & PREVIEW** panel on the left. Click the radio buttons at the top to switch between detection modes (e.g., `Sticker`). The system will automatically swap the YOLO model in the background.

### Step 2: Draw Region of Interest (ROI)
1. Ensure the Transcend ECM camera feed is displaying correctly on the left panel.
2. **Click and drag** your mouse over the left video feed to draw a red bounding box (ROI).
3. Release the mouse button. The system will lock onto this area and begin continuous automatic detection.

As shown in the image below, once the target enters the user-defined red bounding box (ROI), the system automatically isolates the object, physically deskews (straightens) it, and begins defect detection.

![ROI Selection](./Picture1.png)

### Step 3: Review Detection Results
The right panel provides a detailed Gallery Inspection view. The system provides clear visual feedback based on the detection results.

**PASS State:**
If no defects are detected, the zoomed inspection gallery will display a green glowing border along with a clear **PASS** indicator.

![PASS Result](./Picture3.png)

**FAIL State:**
If defects are found, the inspection gallery will switch to a red glowing border. The exact locations of the defects will be highlighted with red bounding boxes directly on the deskewed image.

![FAIL Result](./Picture4.png)

* **Mouse Wheel Zoom**: Hover over the right panel and use your **mouse wheel** to zoom in and out for closer manual inspection.
* **OCR Readout**: The bottom right section automatically formats and displays the text recognized by PaddleOCR.

### Step 4: Collect Training Data
If you encounter a false positive or a missed detection and want to add it to your dataset for future training:
* Ensure the target object is currently displayed in the right Gallery panel.
* Press the **`S`** key on your keyboard.
* The status bar will display "Saved X sticker sample(s)...".
* The cropped images and corresponding YOLO `.txt` label files will be automatically saved in the `dataset_update/` directory.

### Step 5: Toggle View or Exit
* Press the **`ESC`** key to toggle between full-screen and windowed modes.
* Click the `X` button on the window (in windowed mode) to close the application. The system will safely terminate background workers and release camera/GPU resources.

---

## Troubleshooting

1. **Q: The app crashes immediately, and the terminal shows "Cannot load icon".**
   * **A**: Ensure you have an image file named `ts.png` in the root directory of the project.

2. **Q: The bottom status bar is stuck at "Detection model not found: stick".**
   * **A**: Make sure you have created the `models` folder and placed the `stick_best.pt` file inside it.

3. **Q: The screen is black, and the terminal shows a critical error about not finding a camera device.**
   * **A**: 
     1. Check your Transcend ECM camera's connection.
     2. The program scans `/dev/v4l/by-path/` and `/dev/video*` (Linux) or index 0-5 (Windows). If your camera is being used by another application (like OBS or a web browser), close that application and restart this script.

4. **Q: PaddleOCR throws an error on Linux (missing libgomp or similar libraries).**
   * **A**: Install the required system dependencies by running: 
     `sudo apt-get install libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev`