from pathlib import Path

import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Use image 2 as the reference image and image 1 as the target image.
refer_image = Path("chair/image.png")
target_image = Path("chair/chair8.jpg")
output_path = target_image.with_name(f"{target_image.stem}_vp_result.jpg")

# Initialize a YOLOE model
model = YOLOE("yoloe-26n-seg.pt")

# Bounding boxes on demopic/2.jpg that cover the two monitors.
# Format: [x1, y1, x2, y2] in the original pixel coordinates of the reference image.
visual_prompts = dict(
    bboxes=np.array(
        [
          
            [161.74, 55.12, 661.19, 707.61]# right monitor in 2.jpg
        ],
        dtype=np.float32,
    ),
    cls=np.array([0], dtype=np.int32),
)

# Extract visual prompt embeddings from image 2, then detect similar objects in image 1.
results = model.predict(
    source=str(target_image),
    refer_image=str(refer_image),
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    conf=0.05,
    imgsz=640,
)

# Show and save results.
results[0].show()
results[0].save(filename=str(output_path))
print(f"Saved result to: {output_path.resolve()}")
