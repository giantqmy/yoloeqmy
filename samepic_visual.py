from pathlib import Path

#同一张图中的一个目标作为提示
import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Input and output paths
image_path = Path("demopic/2.jpg")
output_path = image_path.with_name(f"{image_path.stem}_samepic_result.jpg")

# Initialize a YOLOE model
model = YOLOE("yoloe-26n-seg.pt")

# Define visual prompts using bounding boxes and their corresponding class IDs.
# Each box highlights an example of the object you want the model to detect.
visual_prompts = dict(
    bboxes=np.array(
        [
            # [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
            # [120, 425, 160, 445],  # Box enclosing glasses
            [1456.69, 1105.08, 3022.65, 2263.56]

        ],
    ),
    cls=np.array(
        [
            0,  # ID to be assigned for person
            # 1,  # ID to be assigned for glasses
        ]
    ),
)

# Run inference on an image, using the provided visual prompts as guidance
results = model.predict(
    str(image_path),
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
)

# Show and save results
results[0].show()
results[0].save(filename=str(output_path))
print(f"Saved result to: {output_path.resolve()}")
