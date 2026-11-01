"""Run models/cards.onnx on photos and print the cards found, the same way the
app does it (scale to 640 with grey bars, best class per box, drop overlaps).

    pip install onnxruntime pillow numpy
    python tools/try_model.py photo1.jpg photo2.jpg --save out/
"""
import argparse
import pathlib

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps

LABELS = ["10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S",
          "4C", "4D", "4H", "4S", "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S",
          "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S", "9C", "9D", "9H", "9S",
          "AC", "AD", "AH", "AS", "JC", "JD", "JH", "JS", "KC", "KD", "KH", "KS",
          "QC", "QD", "QH", "QS"]
SIZE = 640


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    r, bt = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, r - x) * max(0, bt - y)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter or 1)


def detect(session, image, threshold):
    w, h = image.size
    k = SIZE / max(w, h)
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(image.resize((round(w * k), round(h * k))), (0, 0))
    x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255
    out = session.run(None, {"images": x})[0][0]
    scores = out[4:]
    best, cls = scores.max(0), scores.argmax(0)
    found = []
    for i in np.where(best >= threshold)[0]:
        cx, cy, bw, bh = out[:4, i] / k
        found.append((float(best[i]), LABELS[cls[i]], (cx - bw / 2, cy - bh / 2, bw, bh)))
    found.sort(reverse=True)
    kept = []
    for f in found:
        if all(iou(f[2], o[2]) <= 0.5 for o in kept):
            kept.append(f)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model", default="models/cards.onnx")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--save", help="folder to write pictures with boxes drawn on them")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model)
    for path in args.images:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image.thumbnail((1280, 1280))
        found = detect(session, image, args.threshold)
        print(path, ", ".join(f"{name} {score:.0%}" for score, name, _ in found) or "nothing found")
        if args.save:
            draw = ImageDraw.Draw(image)
            for score, name, (x, y, w, h) in found:
                draw.rectangle((x, y, x + w, y + h), outline=(255, 210, 31), width=3)
                draw.text((x, y - 12), f"{name} {score:.0%}", fill=(255, 210, 31))
            out = pathlib.Path(args.save)
            out.mkdir(parents=True, exist_ok=True)
            image.save(out / pathlib.Path(path).name)


if __name__ == "__main__":
    main()
