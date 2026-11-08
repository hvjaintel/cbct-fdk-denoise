# Models

This sample does **not** ship trained clinical models.

## Denoise stub (`denoise_stub/`)

Tiny OpenVINO smoothing CNN for path validation:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_stub_ir.py
```

## RED-CNN (`red_cnn/`)

Image-domain RED-CNN demo IR (synthetic train → OpenVINO):

```powershell
.\.venv\Scripts\pip.exe install -r requirements-redcnn.txt
.\.venv\Scripts\python.exe .\scripts\train_export_red_cnn.py
```

See [red_cnn/README.md](red_cnn/README.md).

`auto` denoise order: **RED-CNN → stub → classical**.
