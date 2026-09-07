from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# =========================
# PATH SETTINGS
# =========================

IMAGES_DIR = Path(r"C:\Users\tahas\Pictures\link din post\Make dataset with blender\outpouts\images")
LABELS_DIR = Path(r"C:\Users\tahas\Pictures\link din post\Make dataset with blender\outpouts\labels")
CLASSES_FILE = Path(r"C:\Users\tahas\Pictures\link din post\Make dataset with blender\outpouts\classes.txt")

OUTPUT_DIR = Path(r"C:\Users\tahas\Pictures\link din post\Make dataset with blender\outpouts\drawed lables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD CLASS NAMES
# =========================

if CLASSES_FILE.exists():
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
else:
    classes = []

print("Classes:", classes)


# =========================
# DRAW FUNCTION
# =========================

def draw_yolo_labels(image_path, label_path, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    if not label_path.exists():
        print(f"No label found for: {image_path.name}")
        image.save(output_path)
        return

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()

        if len(parts) < 5:
            print(f"Invalid label line in {label_path.name}: {line}")
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # YOLO normalized coordinates -> pixel coordinates
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)

        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # Keep coordinates inside image
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        # Class name
        if 0 <= class_id < len(classes):
            class_name = classes[class_id]
        else:
            class_name = f"class_{class_id}"

        # Bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=4
        )

        # Label background
        text = f"{class_name} [{class_id}]"

        try:
            text_bbox = draw.textbbox((x1, y1), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            text_width = len(text) * 8
            text_height = 15

        label_y = max(0, y1 - text_height - 8)

        draw.rectangle(
            [
                (x1, label_y),
                (x1 + text_width + 8, y1)
            ],
            fill="red"
        )

        draw.text(
            (x1 + 4, label_y + 2),
            text,
            fill="white"
        )

    image.save(output_path, quality=95)


# =========================
# PROCESS ALL IMAGES
# =========================

image_extensions = {".jpg", ".jpeg", ".png"}

image_files = sorted([
    p for p in IMAGES_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
])

print(f"Found {len(image_files)} images")

processed = 0

for image_path in image_files:

    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    output_path = OUTPUT_DIR / image_path.name

    draw_yolo_labels(
        image_path=image_path,
        label_path=label_path,
        output_path=output_path
    )

    processed += 1

    print(f"[{processed}/{len(image_files)}] Saved: {output_path.name}")


print("\nDone!")
print("Output folder:")
print(OUTPUT_DIR)