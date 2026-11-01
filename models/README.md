# cards.onnx

YOLOv8n object detector that finds the rank-and-suit index in the corners of playing cards. It has 52 classes (`10C`, `AH`, `QS`, …; the order is in `detector.js`). Input: `images`, 1×3×640×640 RGB in 0–1, letterboxed with grey (114). Output: `output0`, 1×56×8400 (box centre x, y, width, height, then one score per class).

Source: [mustafakemal0146/playing-cards-yolov8](https://huggingface.co/mustafakemal0146/playing-cards-yolov8) on Hugging Face, file `playing_cards_model_0_playing-cards-colab.pt`. It was trained on the Roboflow *Playing Cards* dataset. It was converted with `tools/export_model.py`, which never runs code from the downloaded file.

License of the model: MIT, as published by its author on Hugging Face.

Why this one: on 10 real phone photos of single cards (French deck, from the MIT-licensed `drFarid/French-Playing-Cards` dataset) it named 8 correctly, while a YOLO11n model trained on the same dataset named 6. The misses were a jack printed with a French *V* and a card held against a bright window.

To replace it, export another model with `tools/export_model.py`, update `LABELS` in `detector.js` if the class order differs, and change `MODEL_CACHE` in `app.js` and `sw.js` so phones download the new file.
