from pathlib import Path

from PIL import Image, ImageDraw


image_path = Path(r"D:\anaconda\envs\yoloe26\lib\site-packages\ultralytics\assets\bus.jpg")
output_path = Path(r"D:\Desktop\codeshixi\bus_prompt_boxes.jpg")

# The same boxes used in samepic_visual.py
bboxes = [
    [221.52, 405.8, 344.98, 857.54],
    [120.0, 425.0, 160.0, 445.0],
]
labels = ["cls=0", "cls=1"]
colors = ["red", "lime"]


image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

for bbox, label, color in zip(bboxes, labels, colors):
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
    text_box = (x1, max(0, y1 - 22), x1 + 70, y1)
    draw.rectangle(text_box, fill=color)
    draw.text((x1 + 4, max(0, y1 - 20)), label, fill="black")

image.save(output_path, quality=95)
image.show()
print(f"Saved visualization to: {output_path}")
