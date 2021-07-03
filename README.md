# 046211 Project - Image Captioning
#### Python implementation of Image Captioning networks
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/dog.png) ![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/kayak.png)

The project uses PyTorch to implement two networks that generate descriptive sentences from an image input. The models were inspired by a couple of existing projects[1, 2].

## Dataset
The networks were trained using the Flickr8k dataset

## Model Architectures

### **Model 1**
CNN-RNN with Single layer LSTM without Attention
*   **Encoder** - Pre-Trained Inception v3
*   **Decoder** - Single Layer LSTM

### **Model 2**
CNN-RNN with Single layer LSTM with Attention
*   **Encoder** - Pre-Trained ResNet-50
*   **Decoder** - Single Layer LSTM with Soft Attention

## Examples
### **Model 1**
Below is an example of the generated caption:\
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/boy.png)

### **Model 2**
Below is an example of the generated caption and the attention weights:
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/attention_motorcyclist.png)


## Scores

# Usage
```python
from PencilDrawingBySketchAndTone import *
import matplotlib.pyplot as plt
ex_img = io.imread('./inputs/11--128.jpg')
pencil_tex = './pencils/pencil1.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=0, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                       stroke_darkness= 2,tone_darkness=1.5)
plt.rcParams['figure.figsize'] = [16,10]
plt.imshow(ex_im_pen)
plt.axis("off")
```
Check out the notebook for additional information.
# Parameters
* kernel_size = size of the line segement kernel (usually 1/30 of the height/width of the original image)
* stroke_width = thickness of the strokes in the Stroke Map (0, 1, 2)
* num_of_directions = stroke directions in the Stroke Map (used for the kernels)
* smooth_kernel = how the image is smoothed (Gaussian Kernel - "gauss", Median Filter - "median")
* gradient_method = how the gradients for the Stroke Map are calculated (0 - forward gradient, 1 - Sobel)
* rgb = True if the original image has 3 channels, False if grayscale
* w_group = 3 possible weight groups (0, 1, 2) for the histogram distribution, according to the paper (brighter to darker)
* pencil_texture_path = path to the Pencil Texture Map to use (4 options in "./pencils", you can add your own)
* stroke_darkness = 1 is the same, up is darker.
* tone_darkness = as above

# Folders
* Examples: a few sample images from the Flickr30k dataset.

# References
[1] https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning

[2] https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch
