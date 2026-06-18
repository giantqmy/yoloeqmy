from pathlib import Path

from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-26n-seg-pf.pt")

image_path = Path("demopic/2.jpg")
output_path = image_path.with_name(f"{image_path.stem}_free_prompt_result.jpg")

# Run prediction. No prompts required.
results = model.predict(str(image_path))

# Show and save results
results[0].show()
results[0].save(filename=str(output_path))
print(f"Saved result to: {output_path.resolve()}")
