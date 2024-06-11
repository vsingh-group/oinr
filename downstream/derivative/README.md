This directory contains code to compute the derivative of a 2d image using a pre-trained O-INR of the image in one forward pass efficiently. We include the script for fitting the O-INR on the 2d image to make it complete.

The first step is to train the O-INR on the image. One could use the command below. Please check the arguments for additional choice.

```
python main_discrete.py --img_path "path to image"
```

The file main_gradinet.py contains code to compute the gradient of an image from its O-INR representation. The model.py file has the models defined for both O-INR and it's derivative forward pass. 

Once the O-INR is trained, the following command can be used to compute the gradient of the image.

```
python main_gradient.py --ckpt_path "path to final model check point"
```
