# VisionDesk_assignment (Canny Edge Detector)

A Python-based application that provides real-time camera feed processing with Canny Edge Detection and multiple image filtering capabilities using OpenCV and PyQt5.

Video Drive Link: https://drive.google.com/file/d/1pfcMe0wyqm80hmZnb1raQfFMwbDH8MN0/view?usp=sharing

## Features

### Core Functionality
- **Live Camera Feed** - View real-time video from your webcam
- **Canny Edge Detection** - Toggle between edge detection and normal view
- **Threshold Control** - Adjust low and high thresholds for edge detection in real-time
- **Region of Interest (ROI)** - Select specific areas to apply processing

### Additional Features
- **Multiple Filters** - Apply various image filters:
  - Grayscale
  - Sepia
  - Blur
  - Sharpen
  - Invert
- **Capture Options**
  - Take snapshots (saved as PNG/JPG)
  - Record video (saved as AVI)
- **Performance Metrics** - Real-time FPS counter and resolution display

## How It Works

### Main Components

The application architecture consists of several key components:

1. **Main Window**: Contains the video display and control panels
2. **Video Capture**: Uses OpenCV to access the webcam
3. **Processing Pipeline**: Applies selected filters to the webcam feed
4. **PyQt5 Interface**: Provides UI controls and displays processed images

### Processing Workflow

1. **Image Acquisition**: Captures frames from the webcam
2. **ROI Handling**: If enabled, extracts the selected region of interest
3. **Image Processing**: Applies either Canny Edge Detection or selected filter
4. **Display Update**: Converts processed image to QImage format and displays it
5. **Recording/Snapshot**: Saves processed frames if recording or snapshot is requested

### Code Structure

- **ImageProcessingApp Class**: Main application class that handles UI and processing
- **Setup Methods**: Initialize the UI components and camera connection
- **Event Handlers**: Process user interactions (button clicks, slider movements)
- **Processing Functions**: Implement image filters and edge detection
- **Utility Methods**: Handle file operations and performance tracking

## Performance Considerations

- The application displays current FPS (frames per second)
- Complex filters and high-resolution cameras may reduce FPS
- Recording video requires additional processing power

## Customization

The application can be extended with:

- Additional filters by adding new entries to the `apply_filter` method
- Support for multiple cameras by modifying the camera index
- Custom UI themes by updating the stylesheet in the main function

## Acknowledgments

- OpenCV for image processing capabilities
- PyQt5 for the UI framework

### Help From
- Calude.ai
