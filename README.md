# neural-style-transfer
Repository to apply neural style transfer on content and style image to create new artistic images.  
Paper: [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
# Dependencies: Python3
```
tensorflow==1.9.0
keras==2.2.4
numpy==1.15.4
PIL==4.2.1
scipy==1.1.0
matplotlib==2.2.2
```

# Execution: tbd
The project has following directory structure:
root:
|  
|-- content: containing all content images  
|-- style: containing all style images  
|-- output: containing all subdirectories for generated images  
|`main.py`  

The python main programm `main.py` has following input arguments:  
**Required input arguments**
* `-content_image_path`: This variable stores the filename of the content image. Note you do not have to add the `conten/` subdirectory. E.g `content_image_path = focus_left.jpg` 
* `-style_image_path`: This variable stores the filename of the style image. Note you do not have to add the `style/` subdirectory. E.g `style_image_path = hokusai-a_colored_version_of_the_big_wave.jpg`  
  
**Optional input arguments**
* `-output_subdir`: This variable creates a subdirectory in the output directory, hence a new directory `output/output_subdir` will be created. Default value is to select the basename of the content image.
* `-init_image`: Which image to select as initial generated image. Choices are `content` or `random`.  
Note with `random` you might need more iterations. Default is `content`.
* `-image_width`: Width of the generated image. By default this value is `600`.
* `-image_height`: Height of the generated image. By default this value is `600`.
* `-content_layer`: Which layer of either `VGG16` or `VGG19` to use as content layer. By default this value is `block5_conv2`.
* `-style_layers`: Which layers of either `VGG16` or `VGG19` to use as style layers. By default this value is ```['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']```.
* `-content_weight`: Weight of content loss. By default this value is `0.025`.
* `-style_weights`: Weight for style loss. Note since the style loss will be computed through several `style_layers`. Let `n` be the number of style layers, hence for each layer one can apply a different style weight. By default the `style_weights` is ```python style_weights = [1.] * n```. If you want to adjust different weights for the style layers type in `-style_weights w1 w2 w3 w4 .. wn` in the python shell. Note that the number of inputted weights must match the number of  style layers `n`.
* `-total_variation_weight`: Weight for the third loss. By default this value is `8.5e-5`.
* `-num_iter`: Number of iterations to run the algorithm. By default this value is `20`.
* `-model`: Which model to select in order to used the pre-trained weights. By default this is `VGG16`. Choices are `VGG16` and `VGG19`.
* `-rescale_image`: Whether or not to rescale the generated image to the size of the content image. Default is `False`.

**Example shell calls**
1) Use all default values:  
`python main.py -content_image_path focus_left.jpg -style_image_path = hokusai-a_colored_version_of_the_big_wave.jpg`
2) Use different content layer and different style layers with increasing style weights on 10 iterations with VGG19:  
`python main.py -content_image_path focus_left.jpg -style_image_path hokusai-a_colored_version_of_the_big_wave.jpg -num_iter 10 -model VGG19 -content_layer block4_conv2 -content_weight -style_layers block1_conv1 block2_conv1 block3_conv1 -style_weights 4.0 8.0 100.0`
  
**Layers**  
In order to have a look at the model summary for `VGG16` and `VGG19` have a look at this [Notebook](https://github.com/ptl93/conv-net-feature-extraction/blob/master/feature-extraction.ipynb)

# Result:
## Content Image:
Following content image was selected for the algorithm:
![Content Image](https://github.com/ptl93/neural-style-transfer/blob/master/content/focus_left.jpg)
## Style Image:
Following style image (Hokusai's "Colored Big Wave") was selected for the algorithm:
![Style Image](https://github.com/ptl93/neural-style-transfer/blob/master/style/hokusai-a_colored_version_of_the_big_wave.jpg)
## Result:
This image was generated after 60 iterations. Note 10 iterations would have been sufficient.
![Generated Image](https://github.com/ptl93/neural-style-transfer/blob/master/output/focus_left/generated_image_at_iteration_60.png)
### How the algorithm adjusts the image:
![Animation of algorithm](https://github.com/ptl93/neural-style-transfer/blob/master/output/focus_left/generated_wave_gif.gif)

## Loss Curve:
![Loss Curve](https://github.com/ptl93/neural-style-transfer/blob/master/output/focus_left/loss_history.jpg)
