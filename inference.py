import argparse
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
#from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="ALPR inference with bounding box visualization and OCR"
    )
    parser.add_argument(
        "--model", required=True, help="Path to trained YOLO .pt weights"
    )
    parser.add_argument("--img", required=True, help="Path to input image")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (default: 640)"
    )
    parser.add_argument("--viz", action="store_true", help="Visualize bounding boxes")
    parser.add_argument(
        "--no-ocr", dest="ocr", action="store_false", help="Disable OCR"
    )
    args = parser.parse_args()
    return args


def load_image(path):
    try:
        return np.array(Image.open(path).convert("RGB"))
    except FileNotFoundError:
        print(f"Image not found: {path}", file=sys.stderr)
        sys.exit(1)


def load_model(path):
    try:
        return YOLO(path, task="detect")
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)


def detect(img, model, img_size, confidence_threshold):
    results = model(img, imgsz=img_size, conf=confidence_threshold, verbose=False)
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confidences = r.boxes.conf.cpu().numpy()
    results = []
    for box, c, score in zip(boxes, classes, confidences):
        # [class_name, confidence, x1, y1, x2, y2]
        results.append((model.names[c], float(score), *box.tolist()))
    return results


def crop_bbox(img, coords):
    x1, y1, x2, y2 = map(int, coords)
    return img[y1:y2, x1:x2]


def visualize(img, detections):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    for d in detections:
        class_name, conf, x1, y1, x2, y2 = d
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"{class_name} {conf:.2f}", color="red", fontsize=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# def load_ocr_model():
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
#     return processor, model


def ocr(img, processor, ocr_model):
    img_pil = Image.fromarray(img)
    pixel_values = processor(images=img_pil, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    args = parse_args()

    img = load_image(args.img)
    model = load_model(args.model)

    detections = detect(img, model, img_size=args.imgsz, confidence_threshold=args.conf)
    # Detection output: [class_name, confidence, x1, y1, x2, y2]
    for d in detections:
        print(f"Class: {d[0]}, Confidence: {d[1]:.2f}, Box: {d[2:]}")

    if args.viz:
        visualize(img, detections)

    # if args.ocr:
    #     processor, ocr_model = load_ocr_model()
    #     for d in detections:
    #         class_name, conf, x1, y1, x2, y2 = d
    #         crop = crop_bbox(img, (x1, y1, x2, y2))
    #         text = ocr(crop, processor, ocr_model)
    #         print(f"Plate ({class_name}, {conf:.2f}): {text if text else '<no text>'}")


if __name__ == "__main__":
    main()