# Nametag

A phone camera that sticks name labels on people and things, and remembers where it last saw them and who they were with.

Open **https://adamecytb-stack.github.io/nametag/** in Safari on an iPhone, tap Share, then **Add to Home Screen**. Open it from the home screen icon from then on: the home screen app keeps its own storage, separate from Safari.

## How it works

- Point the back camera at something. Unnamed people show `who?`, unnamed things show what they are (`cup`, `backpack`).
- Tap one and give it a name. That is the training: one example is enough to start, and "Add this view as another example" makes it better.
- Named people get a yellow label, named things a white one.
- Tap a label to see where and when it was last seen and who it was with, and to add a note.
- **Photo** teaches it from pictures already in your library.
- The label in the top left names the place you are in. With location allowed, it fills in by itself next time you are there.

## Where things run

Everything runs on the phone. No photo, face or sighting is uploaded anywhere, and this repository holds only the app's code. Names, examples and sightings are stored in the browser's IndexedDB; Settings has a backup file export and restore.

| Job | Model |
| --- | --- |
| Finding people and things | MediaPipe EfficientDet-Lite0 (80 everyday object types) |
| Telling people apart | face-api tiny face detector, 68-point landmarks, 128-number face descriptor |
| Telling things apart | MediaPipe MobileNetV3 image embedding, compared by cosine similarity |

The models download from jsDelivr and Google's MediaPipe bucket on first launch (about 30 MB) and are cached for offline use.

## Run it locally

```bash
python -m http.server 8791
```

Then open http://localhost:8791. The camera needs https or localhost.
