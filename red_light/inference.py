"""
Inference utilities for Red Light Violation Detection.

Expose helpers and `run_inference` for programmatic use; CLI wrappers should
stay thin and delegate here.
"""

from pathlib import Path
import cv2
from ultralytics import YOLO

from red_light.config import (
    ConfigError,
    load_data_config,
    load_inference_config,
    validate_data_yaml_path,
    validate_model_path,
    validate_source_path,
)


def get_cv2_font(font_name):
    """Map font name string to OpenCV constant."""
    font_map = {
        'FONT_HERSHEY_SIMPLEX': cv2.FONT_HERSHEY_SIMPLEX,
        'FONT_HERSHEY_PLAIN': cv2.FONT_HERSHEY_PLAIN,
        'FONT_HERSHEY_DUPLEX': cv2.FONT_HERSHEY_DUPLEX,
        'FONT_HERSHEY_COMPLEX': cv2.FONT_HERSHEY_COMPLEX,
        'FONT_HERSHEY_TRIPLEX': cv2.FONT_HERSHEY_TRIPLEX,
        'FONT_HERSHEY_COMPLEX_SMALL': cv2.FONT_HERSHEY_COMPLEX_SMALL,
        'FONT_HERSHEY_SCRIPT_SIMPLEX': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        'FONT_HERSHEY_SCRIPT_COMPLEX': cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    }
    return font_map.get(font_name, cv2.FONT_HERSHEY_SIMPLEX)


def draw_detections(image, results, class_names, viz_config):
    """Draw detection boxes on image."""
    img = image.copy()

    class_colors = viz_config['class_colors']
    default_color = tuple(class_colors['default'])

    bbox_thickness = viz_config['bbox_thickness']
    font = get_cv2_font(viz_config['label_font'])
    font_scale = viz_config['label_font_scale']
    font_thickness = viz_config['label_font_thickness']
    label_bg_offset = viz_config['label_bg_offset']
    label_text_offset = viz_config['label_text_offset']
    label_text_color = tuple(viz_config['label_text_color'])

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            color = tuple(class_colors.get(class_name, default_color))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, bbox_thickness)

            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            cv2.rectangle(
                img,
                (x1, y1 - label_h - label_bg_offset),
                (x1 + label_w, y1),
                color,
                -1
            )

            cv2.putText(
                img,
                label,
                (x1, y1 - label_text_offset),
                font,
                font_scale,
                label_text_color,
                font_thickness
            )

    return img


def infer_image(model, image_path, class_names, conf_threshold, output_dir, save_txt, viz_config):
    """Run inference on a single image."""
    image_path = Path(image_path)
    print(f"\nProcessing: {image_path.name}")

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        verbose=False
    )

    img_with_detections = draw_detections(
        img, results, class_names, viz_config)

    output_path = output_dir / f"result_{image_path.name}"
    cv2.imwrite(str(output_path), img_with_detections)
    print(f"  Saved to: {output_path}")

    total_detections = len(results[0].boxes)
    print(f"  Detections: {total_detections}")

    if total_detections > 0:
        class_counts = {}
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print("  Detected objects:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    - {class_name}: {count}")

    if save_txt:
        txt_path = output_dir / f"result_{image_path.stem}.txt"
        with open(txt_path, 'w') as f:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                f.write(
                    f"{class_name} {confidence:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")


def infer_directory(model, input_dir, class_names, conf_threshold, output_dir, save_txt, viz_config, image_extensions):
    """Run inference on all images in a directory."""
    input_dir = Path(input_dir)

    image_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"\nFound {len(image_files)} images")
    print("=" * 60)

    for image_path in image_files:
        infer_image(model, image_path, class_names, conf_threshold,
                    output_dir, save_txt, viz_config)

    print("=" * 60)
    print(f"\nAll results saved to: {output_dir}")


def run_inference(model_path, source, infer_config_path, save_txt=False):
    """
    Run inference with provided model and source using configs.

    Args:
        model_path: path to model weights
        source: path to image or directory
        infer_config_path: path to inference configuration YAML
        save_txt: whether to save detection results as text files
    """
    infer_config = load_inference_config(infer_config_path)
    inference_params = infer_config['inference']
    viz_config = infer_config['visualization']

    data_yaml = validate_data_yaml_path(inference_params['default_data_yaml'])
    data_config = load_data_config(data_yaml)
    class_names = data_config['names']

    image_extensions = set(inference_params['image_extensions'])

    output_dir = Path(inference_params['default_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    conf_threshold = inference_params['default_conf_threshold']

    model_path = validate_model_path(model_path)

    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    print(f"Confidence threshold: {conf_threshold}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")

    source_path = validate_source_path(source)

    if source_path.is_file():
        infer_image(model, source_path, class_names,
                    conf_threshold, output_dir, save_txt, viz_config)
    elif source_path.is_dir():
        infer_directory(model, source_path, class_names, conf_threshold,
                        output_dir, save_txt, viz_config, image_extensions)
    else:
        raise ValueError(f"{source} is not a valid file or directory")

    print("\nInference completed!")
