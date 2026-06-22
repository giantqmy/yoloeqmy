from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


# Multiple reference images for the same category.
# Add more reference items here. Each item needs:
# - image: reference image path
# - bboxes: one or more xyxy boxes on that reference image
# - cls: class ids for those boxes, sequential from 0
reference_items = [
    {
        "image": Path("chair/chair13.jpg"),
        "bboxes": np.array(
            [
                [417.31, 310.35, 1157.19, 1292.62],
            ],
            dtype=np.float32,
        ),
        "cls": np.array([0], dtype=np.int32),
    },
    {
        "image": Path("chair/chair13.jpg"),
        "bboxes": np.array(
            [
                [373.98, 292.90, 999.79, 1229.17],
            ],
            dtype=np.float32,
        ),
        "cls": np.array([0], dtype=np.int32),
    },
    
    {
        "image": Path("chair/chair11.jpg"),
        "bboxes": np.array(
            [
                [356.32, 86.19, 1123.50, 1275.01],
            ],
            dtype=np.float32,
        ),
        "cls": np.array([0], dtype=np.int32),
    },
    {
        "image": Path("chair/chair12.jpg"),
        "bboxes": np.array(
            [
                [399.09, 231.15, 1017.04, 1267.90],
            ],
            dtype=np.float32,
        ),
        "cls": np.array([0], dtype=np.int32),
    },
    
    
    # Example for a second reference image:
    # {
    #     "image": Path("chair/chair7.jpg"),
    #     "bboxes": np.array(
    #         [
    #             [x1, y1, x2, y2],
    #         ],
    #         dtype=np.float32,
    #     ),
    #     "cls": np.array([0], dtype=np.int32),
    # },
]

target_image = Path("chair/chair8.jpg")
output_dir = Path("chair/vp_results")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{target_image.stem}_multi_ref_vp_result.jpg"

target_classes = ["chair"]
conf = 0.05
imgsz = 640

# Initialize a YOLOE model
model = YOLOE("yoloe-26n-seg.pt")

# Create a predictor that only extracts visual prompt embeddings from reference images.
vp_predictor = YOLOEVPSegPredictor(
    overrides={
        "task": model.model.task,
        "mode": "predict",
        "save": False,
        "verbose": False,
        "batch": 1,
        "imgsz": imgsz,
    },
    _callbacks=model.callbacks,
)
vp_predictor.setup_model(model=model.model)

vpe_list = []
for i, ref in enumerate(reference_items, start=1):
    prompts = {
        "bboxes": ref["bboxes"].copy(),
        "cls": ref["cls"].copy(),
    }
    vp_predictor.set_prompts(prompts)
    vpe = vp_predictor.get_vpe(str(ref["image"]))
    vpe_list.append(vpe)
    print(f"Loaded reference {i}: {ref['image']}")

if not vpe_list:
    raise ValueError("reference_items is empty. Please provide at least one reference image.")

# Merge visual embeddings from all reference images.
merged_vpe = torch.stack(vpe_list, dim=0).mean(dim=0)
print(f"Merged VPE shape: {tuple(merged_vpe.shape)}")

# Set the merged embedding as the class embedding, then run prediction on the target image.
model.set_classes(target_classes, merged_vpe)
results = model.predict(
    source=str(target_image),
    conf=conf,
    imgsz=imgsz,
)

results[0].show()
results[0].save(filename=str(output_path))
print(f"Saved result to: {output_path.resolve()}")
