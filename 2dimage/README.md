This directory contains code to run O-INR on 2d images. Each main file has it's own argument parser, please refer to them for additional choice. Running the default scripts is noted below.

Running, O-INR with standard convolution on 2d image
```
python main_discrete.py --img_path "path to image file"
```

Running, O-INR with continuous convolution on 2d image
```
python main_continue.py --img_path "path to image file"
```

Running, O-INR with standard convolution to denoise 2d image
```
python main_denoise.py --img_path "path to image file"
```
