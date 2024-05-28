import argparse
import cv2
import math
from ultralytics import YOLO
from PIL import Image


def load_model(model_path):
    """
    Load the YOLO model from the given path.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_class_names():
    """
    Returns a list of class names used for detection.
    """
    return ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
            'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
            'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']


def create_video_writer(video_cap, output_filename):
    """
    Create a video writer object to save the processed video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    return writer


def draw_boxes(image, results, class_names):
    """
    Draw bounding boxes and labels on the image based on detection results.
    """
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = class_names[cls]

            if conf > 0.5:
                if current_class in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    color = (0, 0, 255)
                elif current_class in ['Hardhat', 'Safety Vest', 'Mask']:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.putText(image, f'{current_class}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    return image


def resize_frame(frame, width=None, height=None):
    """
    Resize the frame to the given width and height while maintaining the aspect ratio.
    """
    if width is None and height is None:
        return frame

    (h, w) = frame.shape[:2]
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


def main():
    parser = argparse.ArgumentParser(description="PPE Detection in Videos using YOLO")
    parser.add_argument('--video', type=str, default='data/videos/example_video.mp4', help='Path to the input video')
    parser.add_argument('--output_video', type=str, default='output/videos/output_video.mp4', help='Path to save the output video')
    parser.add_argument('--output_gif', type=str, default='output/gifs/output_video.gif', help='Path to save the output GIF')
    args = parser.parse_args()

    model_path = "data/models/PPE_Model.pt"
    video_path = args.video
    output_video_path = args.output_video
    output_gif_path = args.output_gif
    display_width = 1280  # Set the desired width for display
    display_height = 720  # Set the desired height for display

    model = load_model(model_path)
    if model is None:
        return

    class_names = get_class_names()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    writer = create_video_writer(cap, output_video_path)
    frames = []

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        img = draw_boxes(img, results, class_names)

        resized_img = resize_frame(img, width=display_width, height=display_height)
        cv2.imshow("Detected Video", resized_img)
        writer.write(img)

        # Convert BGR image to RGB for GIF
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        frames.append(pil_img)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Save frames as GIF
    if frames:
        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        print(f"Processed GIF saved to: {output_gif_path}")


if __name__ == "__main__":
    main()
