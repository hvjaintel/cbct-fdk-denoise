# RED-CNN (demo)

Image-domain residual encoder–decoder CNN for CT denoising (Chen et al., IEEE TMI 2017), adapted with padding for tiled OpenVINO inference.

## Generate IR (required once)

```powershell
.\.venv\Scripts\pip.exe install -r requirements-redcnn.txt
.\.venv\Scripts\python.exe .\scripts\train_export_red_cnn.py
```

Produces `model.xml` / `model.bin` / `model.onnx` / `red_cnn.pt`.

The default train set is **synthetic Shepp-Logan + Gaussian noise** for a working demo only. Swap in your own LDCT-trained weights for real use (check weight/dataset licenses for commercial redistribution).

## Runtime

```powershell
.\run.ps1 -Denoise redcnn
# or dashboard → Denoise mode → redcnn
# auto prefers RED-CNN when this IR exists
```
