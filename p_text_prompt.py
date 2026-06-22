from pathlib import Path

from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-26n-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
# model.set_classes(["monitor","keyboard", "black mouse","pen","white paper"])
target_classes = ["chair"]
model.set_classes(target_classes)
input_path = Path("chair")
output_dir = input_path / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# Run detection on all images in the folder
results = model.predict(str(input_path))

for result in results:
    image_path = Path(result.path)
    output_path = output_dir / f"{image_path.stem}_result.jpg"
    target_found = False

    # Save results for each image
    result.save(filename=str(output_path))
    print(f"Saved result to: {output_path.resolve()}")

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for i, (bbox, class_id, score) in enumerate(zip(xyxy, cls, conf), start=1):
            class_name = result.names[class_id]
            if class_name in target_classes:
                target_found = True
                x1, y1, x2, y2 = bbox.tolist()
                print(
                    f"{image_path.name} {class_name} {i}: "
                    f"bbox=[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}], "
                    f"conf={score:.4f}"
                )

    if not target_found:
        print(f"{image_path.name}: No target bbox found.")
