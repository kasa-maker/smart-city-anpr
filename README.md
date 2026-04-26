# Smart City ANPR - AI Traffic Monitoring System

A sophisticated AI-powered traffic monitoring system that uses computer vision to detect vehicles, track their movement, and recognize license plates in real-time. Built for smart city infrastructure and traffic management applications.

![Smart City Traffic](https://img.shields.io/badge/Computer%20Vision-OpenCV-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-YOLOv8-green)
![Tracking](https://img.shields.io/badge/Tracking-DeepSORT-orange)
![OCR](https://img.shields.io/badge/OCR-EasyOCR-red)

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities

- **Vehicle Detection**: Real-time detection of cars, bikes, buses, and trucks using YOLOv8
- **Multi-Object Tracking**: DeepSORT algorithm tracks vehicles across frames with unique IDs
- **License Plate Recognition**: OCR-based plate reading with image enhancement
- **Cross-Line Detection**: Counts vehicles crossing a virtual detection line
- **Live Dashboard**: Interactive Streamlit dashboard for data visualization
- **Database Logging**: SQLite database stores all vehicle logs for analysis

### Dashboard Features

- Real-time vehicle count statistics
- Vehicle type distribution charts (Pie chart)
- Detection timeline visualization (Bar chart)
- Complete license plate logs with timestamps
- Video upload and processing interface
- Downloadable processed video output

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Input Stream                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              YOLOv8 Vehicle Detection                        │
│         (car, bike, bus, truck classification)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              DeepSORT Object Tracking                        │
│            (Unique ID assignment per vehicle)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           Cross-Line Detection Logic                         │
│          (Triggers when vehicle crosses line)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│        License Plate OCR (EasyOCR + Image Enhancement)      │
│         (CLAHE, Gaussian Blur, Morphology Operations)        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│          SQLite Database + CSV Export                        │
│              Streamlit Dashboard Display                     │
└─────────────────────────────────────────────────────────────┘
```

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Ultralytics YOLOv8** | State-of-the-art vehicle detection model |
| **DeepSORT** | Real-time multi-object tracking algorithm |
| **EasyOCR** | License plate text recognition |
| **OpenCV** | Video processing and image manipulation |
| **Streamlit** | Interactive web dashboard |
| **SQLite** | Lightweight database for vehicle logs |
| **Plotly** | Interactive data visualizations |
| **Pandas** | Data processing and CSV export |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/kasaamali/smart-city-traffic-anpr.git
cd smart-city-traffic-anpr
```

### Step 2: Install Dependencies

```bash
pip install ultralytics deep-sort-realtime opencv-python easyocr pandas sqlite3 streamlit plotly
```

### Step 3: Download Pre-trained Models

The YOLOv8 model will be downloaded automatically on first run. Place your custom license plate detection model in the `models/` directory:

```
models/
└── plate_model.pt    # Custom trained license plate detection model
```

### Step 4: Create Required Directories

```bash
mkdir -p data/videos database output
```

## Usage

### Method 1: Command Line Processing

Run the main processing script:

```bash
python main.py
```

This will:
1. Process the video file specified in `data/videos/sample_video2.mp4`
2. Generate output video with detections at `output/output_real_plates.mp4`
3. Save vehicle logs to `output/final_logs.csv`
4. Store data in `database/traffic.db`

### Method 2: Interactive Dashboard

Launch the Streamlit web interface:

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501` where you can:
- View real-time statistics and charts
- Upload custom videos for processing
- Download processed output videos
- Export vehicle logs

### Configuration

Edit these variables in `main.py` to customize:

```python
VIDEO_PATH = 'data/videos/your_video.mp4'  # Input video path
OUTPUT_PATH = 'output/your_output.mp4'      # Output video path
LINE_Y = int(height * 0.6)                   # Detection line position (60% from top)
```

## Project Structure

```
smart-traffic-anpr/
├── app/
│   └── dashboard.py          # Streamlit web dashboard
├── data/
│   └── videos/               # Input video files
│       └── sample_video2.mp4
├── database/
│   └── traffic.db            # SQLite database (auto-created)
├── models/
│   ├── plate_model.pt        # License plate detection model
│   └── yolov8n.pt            # YOLOv8 nano vehicle detection model
├── output/
│   ├── final_logs.csv        # Exported vehicle logs
│   ├── output_real_plates.mp4 # Processed video output
│   └── output_dashboard.mp4  # Dashboard-processed video
├── .gitignore
├── main.py                   # Main processing script
└── README.md                 # This file
```

## How It Works

### 1. Vehicle Detection

The system uses YOLOv8 (You Only Look Once v8) nano model for real-time vehicle detection. It identifies four vehicle classes:
- Class 2: Car
- Class 3: Bike
- Class 5: Bus
- Class 7: Truck

### 2. Object Tracking

DeepSORT algorithm assigns unique IDs to each detected vehicle and maintains tracking across frames, even during brief occlusions.

### 3. Cross-Line Detection

A horizontal detection line is drawn at 60% of the frame height. When a vehicle's centroid crosses this line:
- Vehicle is counted in the respective category
- License plate OCR is triggered
- Data is logged to the database

### 4. License Plate Recognition

The OCR pipeline includes:
1. **Vehicle Crop**: Extract vehicle region from frame
2. **Plate Detection**: Use custom YOLO model to locate plate area
3. **Image Enhancement**:
   - Grayscale conversion
   - 4x upscaling for better OCR accuracy
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Gaussian blur for noise reduction
   - Multiple threshold methods (Otsu + Binary)
   - Morphology operations (Close + Open)
4. **Text Recognition**: EasyOCR reads the enhanced plate image
5. **Validation**: Regex filters for valid license plate format

## Output

### Console Output

```
Real OCR processing shuru...
Vehicle 1 crossed - OCR chal raha hai...
Plate: ABC123
Vehicle 2 crossed - OCR chal raha hai...
Plate: UNREADABLE
...
Done!
Total vehicles crossed: 45
```

### CSV Export (final_logs.csv)

| vehicle_id | vehicle_type | plate_number | timestamp |
|------------|--------------|--------------|-----------|
| 1 | car | ABC123 | 14:30:25 |
| 2 | bike | UNREADABLE | 14:30:28 |
| 3 | bus | XYZ789 | 14:30:31 |

### Database Schema (traffic.db)

```sql
CREATE TABLE vehicle_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER,
    vehicle_type TEXT,
    plate_number TEXT,
    timestamp TEXT
);
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

---

**Developed for Smart City Traffic Management Solutions**

For issues and queries, please raise an issue on the GitHub repository.
