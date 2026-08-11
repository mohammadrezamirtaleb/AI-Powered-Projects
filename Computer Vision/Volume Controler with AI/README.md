# AI-Powered Volume Controller

This project is a computer vision-based application that allows you to control your system's volume using simple hand gestures. By tracking the distance between your thumb and index finger in real-time, the application dynamically adjusts the master volume on your Windows machine.

## ✨ Features

- **Real-time Hand Tracking**: Uses Google's MediaPipe framework for fast and accurate hand landmark detection.
- **Intuitive Gesture Control**: Adjust the volume seamlessly by moving your thumb and index finger closer together or further apart.
- **Visual Feedback**: Displays an interactive volume bar and a dynamic volume percentage directly on the video feed.
- **Responsive UI Color Indicators**: The volume bar changes color (Green -> Yellow -> Red) based on the current volume level to give immediate visual context.
- **Smooth Volume Adjustment**: Applies a smoothing algorithm to prevent abrupt volume jumps and ensure a pleasant user experience.

## 🛠️ Prerequisites

- Python 3.7+
- A working webcam
- Windows OS (due to the `pycaw` library dependency for system volume control)

## 📦 Installation

1. Clone this repository or download the source code.
2. Navigate to the project directory:
   ```bash
   cd "Computer Vision/Volume Controler with AI"
   ```
3. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

*Note: Ensure you have the `hand_landmarker.task` model file in the same directory as the source code. This is required by MediaPipe for the vision tasks.*

## 🚀 Usage

Run the main script to start the application:

```bash
python main.py
```

### Controls

- **Increase/Decrease Volume**: Show your hand to the camera. Pinch your thumb and index finger to lower the volume, and spread them apart to increase the volume.
- **Quit**: Press the `q` key on your keyboard to exit the application.

## 📂 Project Structure

- `main.py`: The entry point of the application handling video capture, UI rendering, and bridging the tracker with the volume controller.
- `hand_tracker.py`: Contains the `HandTracker` class which encapsulates the MediaPipe logic for detecting hands, extracting landmarks, and drawing the skeleton.
- `volume_controller.py`: Contains the `VolumeController` class which uses `pycaw` to interact with the Windows audio subsystem and change the master volume.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `hand_landmarker.task`: The pre-trained model asset used by MediaPipe for hand landmark detection.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📄 License

This project is open-source and available under the MIT License.
