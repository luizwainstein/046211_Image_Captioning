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
Below are a few examples of the generated captions:\
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/boy.png)
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/boat.png)
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/pool.png)

### **Model 2**
Below are a few examples of the generated captions and the attention weights:
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/attention_motorcyclist.png)
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/attention_soccer.png)
![](https://github.com/luizwainstein/046211_Image_Captioning/blob/main/Images/attention_climber.png)


## Scores

# Usage
* Fill in the following paths:
```python
path_images="/content/flickr8k/Images" #Dataset Images
path_captions="/content/flickr8k/captions.txt" #Dataset Captions
path_examples="" #Images to caption
path_checkpoints="" #Model checkpoints
```
* Use the following function to caption images:
```python
print_examples(model, device, dataset, path, transform, attention=False, save=False, max_imgs=5, dpi=None)
```
Check out the notebook for additional information.
# Parameters
* `model` = model to evaluate
* `device` = device to use i.e. `device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`
* `dataset` = dataset used for vocabulary
* `path` = directories for the images to caption
* `transform` = transform depending on the model 
* `attention` = `True` for Model 2, `False` for Model 1
* `save` = saves the figures with generated captions
* `max_imgs` = generates captions for only max_imgs pictures from the folder (random)
* `dpi` = resolution for saved figures

# Folders
* Examples: a few sample images from the Flickr30k dataset.

# References
[1] https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning

[2] https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch
