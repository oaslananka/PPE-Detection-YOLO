import argparse
import cv2
import math
from ultralytics import YOLO


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
    return [
        'Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
        'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
        'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader'
    ]


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


def main():
    parser = argparse.ArgumentParser(description="PPE Detection in Images using YOLO")
    parser.add_argument('--image', type=str, default='data/images/example_image.jpg', help='Path to the input image')
    parser.add_argument('--output', type=str, default='output/images/output_image.jpg', help='Path to save the output image')
    args = parser.parse_args()

    model_path = "data/models/PPE_Model.pt"
    image_path = args.image
    output_path = args.output

    model = load_model(model_path)
    if model is None:
        return

    class_names = get_class_names()
    try:
        results = model(image_path, show=True, save=True)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    img = draw_boxes(img, results, class_names)
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to: {output_path}")

    cv2.imshow("Detected Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
