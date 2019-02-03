# neural-style-transfer
Repository to apply neural style transfer on content and style image to create new artistic images.

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
