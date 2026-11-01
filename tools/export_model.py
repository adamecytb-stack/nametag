"""Turn a YOLO card model (.pt) into the models/cards.onnx file the app runs.

A .pt file is a Python pickle, and loading an untrusted pickle can run any
code. So this script first lists what the file wants to load without running
it, then loads it with torch's weights_only loader, which refuses everything
except the PyTorch / Ultralytics layer classes listed below.

    python -m venv .venv
    .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    .venv/bin/pip install ultralytics onnx
    .venv/bin/python tools/export_model.py path/to/best.pt models/cards.onnx

The model used now is MIT licensed:
https://huggingface.co/mustafakemal0146/playing-cards-yolov8
(file playing_cards_model_0_playing-cards-colab.pt)
"""
import argparse
import importlib
import json
import pickletools
import zipfile

ALLOWED = {
    "collections.OrderedDict", "builtins.set", "__builtin__.set",
    "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_parameter",
    "torch.FloatStorage", "torch.HalfStorage", "torch.LongStorage", "torch.Size",
    "torch.nn.modules.activation.SiLU", "torch.nn.modules.batchnorm.BatchNorm2d",
    "torch.nn.modules.container.ModuleList", "torch.nn.modules.container.Sequential",
    "torch.nn.modules.conv.Conv2d", "torch.nn.modules.linear.Identity",
    "torch.nn.modules.pooling.MaxPool2d", "torch.nn.modules.upsampling.Upsample",
    "ultralytics.nn.modules.block.Attention", "ultralytics.nn.modules.block.Bottleneck",
    "ultralytics.nn.modules.block.C2f", "ultralytics.nn.modules.block.C2PSA",
    "ultralytics.nn.modules.block.C3k", "ultralytics.nn.modules.block.C3k2",
    "ultralytics.nn.modules.block.DFL", "ultralytics.nn.modules.block.PSABlock",
    "ultralytics.nn.modules.block.SPPF", "ultralytics.nn.modules.conv.Concat",
    "ultralytics.nn.modules.conv.Conv", "ultralytics.nn.modules.conv.DWConv",
    "ultralytics.nn.modules.head.Detect", "ultralytics.nn.tasks.DetectionModel",
}
# Handled by torch itself, not imported by name.
TORCH_INTERNAL = {"torch.FloatStorage", "torch.HalfStorage", "torch.LongStorage", "torch.Size",
                  "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_parameter",
                  "collections.OrderedDict", "builtins.set", "__builtin__.set"}


def globals_in(path):
    """Names the pickle inside `path` would import, found without running it."""
    found = set()
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".pkl"):
                continue
            strings = []
            for op, arg, _ in pickletools.genops(z.read(name)):
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                    strings.append(arg)
                elif op.name == "STACK_GLOBAL":
                    found.add(f"{strings[-2]}.{strings[-1]}")
                elif op.name == "GLOBAL":
                    found.add(arg.replace(" ", "."))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    unexpected = globals_in(args.weights) - ALLOWED
    if unexpected:
        raise SystemExit(f"Refusing to load, the file wants: {sorted(unexpected)}")

    import torch

    safe = [set]
    for name in ALLOWED - TORCH_INTERNAL:
        module, cls = name.rsplit(".", 1)
        safe.append(getattr(importlib.import_module(module), cls))
    torch.serialization.add_safe_globals(safe)

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = (ckpt.get("ema") or ckpt["model"]).float().eval().fuse()
    head = model.model[-1]
    head.export, head.format, head.dynamic = True, "onnx", False
    for p in model.parameters():
        p.requires_grad = False

    x = torch.zeros(1, 3, args.size, args.size)
    print("output shape:", tuple(model(x).shape))  # (1, 4 + classes, 8400)
    torch.onnx.export(model, x, args.out, opset_version=13,
                      input_names=["images"], output_names=["output0"], dynamo=False)
    names = [model.names[i] for i in range(len(model.names))]
    print("classes, in the order detector.js LABELS must use:")
    print(json.dumps(names))


if __name__ == "__main__":
    main()
