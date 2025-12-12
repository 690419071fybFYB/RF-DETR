from pathlib import Path

import supervision as sv
from PIL import Image
from inference import get_model

IMAGE_DIR = Path("/home/fyb/datasets/RSOD_cocoFormat/test2017")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = get_model("rfdetr-base")

image_paths = sorted(
    [
        path
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for path in IMAGE_DIR.glob(ext)
    ]
)

if not image_paths:
    raise FileNotFoundError(f"在 {IMAGE_DIR} 未找到任何支持的图片文件")

box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW)
label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW)

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    predictions = model.infer(image, confidence=0.5)[0]
    detections = sv.Detections.from_inference(predictions)
    labels = [prediction.class_name for prediction in predictions.predictions]

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)

    output_path = OUTPUT_DIR / f"{image_path.stem}_annotated{image_path.suffix}"
    annotated_image.save(output_path)
    print(f"{image_path.name} 结果已保存: {output_path}")
