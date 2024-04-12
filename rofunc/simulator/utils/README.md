# YCB dataset download and URDF generation

## Downlaod the YCB dataset
```bash
python ycb_downloader.py
```

## Generate URDF files
```bash
pip install lxml
python ycb2urdf.py
```

Note: the texture file name in each `textured.mtl` has a weird space character at the end. You need to remove them manually.