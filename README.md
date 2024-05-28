
# PPE-Detection-YOLO

This repository contains a project that uses YOLO for detecting Personal Protective Equipment (PPE) in images and videos. The project includes pre-trained models, scripts for image and video detection, and examples for testing.

## Project Structure

```
PPE-Detection/
├── data/
│   ├── images/
│   ├── videos/
│   └── models/
├── src/
│   ├── image_detection.py
│   ├── video_detection.py
├── output/
│   ├── images/
│   ├── videos/
│   └── gifs/
├── notebooks/
│   ├── EDA.ipynb
├── README.md
├── requirements.txt
├── .gitignore
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oaslananka/PPE-Detection-YOLO.git
   cd PPE-Detection
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Image Detection

Run the following command to detect PPE in an image:
```bash
python src/image_detection.py --image path/to/your/image.jpg --output path/to/save/output_image.jpg
```

### Video Detection

Run the following command to detect PPE in a video:
```bash
python src/video_detection.py --video path/to/your/video.mp4 --output_video path/to/save/output_video.mp4 --output_gif path/to/save/output_video.gif
```

## Example Output

Here is an example output of the video detection saved as a GIF:

![Example Output](output/gifs/output_video.gif)

## Project Components

### data/
Contains sample images, videos, and the pre-trained model.

### src/
Includes the main scripts for image and video detection.

### output/
Stores the processed output images, videos, and GIFs.

### notebooks/
Holds Jupyter notebooks for data exploration and analysis.

## License

This project is licensed under the MIT License.
