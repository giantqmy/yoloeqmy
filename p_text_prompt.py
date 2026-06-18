from pathlib import Path

from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-26n-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
# model.set_classes(["monitor","keyboard", "black mouse","pen","white paper"])
model.set_classes(["monitor","keyboard", "black mouse","pen","white paper"])
image_path = Path("demopic/2_same.jpg")
output_path = image_path.with_name(f"{image_path.stem}_result.jpg")

# Run detection on the given image
results = model.predict(str(image_path))

# Show and save results
results[0].show()
results[0].save(filename=str(output_path))
print(f"Saved result to: {output_path.resolve()}")

monitor_found = False
boxes = results[0].boxes
if boxes is not None and len(boxes) > 0:
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    for i, (bbox, class_id, score) in enumerate(zip(xyxy, cls, conf), start=1):
        class_name = results[0].names[class_id]
        if class_name == "monitor":
            monitor_found = True
            x1, y1, x2, y2 = bbox.tolist()
            print(
                f"monitor {i}: "
                f"bbox=[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}], "
                f"conf={score:.4f}"
            )

if not monitor_found:
    print("No monitor bbox found.")
