# -*- coding: utf-8 -*-
"""
@title: main.py
@author: Tuan Le
@email: tuanle@hotmail.de
"""
################################ Load Modules #################################
from __future__ import print_function
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import os
from keras import backend as K
import sys
###############################################################################


######################### Info and Argument Functions #########################
def args_parser():
    parser = argparse.ArgumentParser(description="Neural style transfer with Keras.")
    
    parser.add_argument("-content_image_path", metavar="", type=str,
                        help="Filename with extension of the content image to transform with style transfer.")
    
    parser.add_argument("-style_image_path", metavar="", type=str,
                        help="Filename with extension of the style image")
    
    parser.add_argument("-output_subdir", metavar="", type=str, default=None,
                    help="Name of output subdir. Default is to create a subdirectory 'output/content_file/'")
    
    parser.add_argument("-init_image", default="content", type=str,
                        help="Initial image used to generate the final image. Options are 'content', or 'random'. Default is 'content'")
    
    parser.add_argument("-image_width", default=600, metavar="", type=int,
                        help="Width of generated image. Default is 600")
    
    parser.add_argument("-image_height", default=600, metavar="", type=int,
                        help="Height of generated image. Default is 600")
    
    parser.add_argument("-content_layer", metavar="", default="block5_conv2", type=str,
                        help="Content layer used for content loss. Default is 'block5_conv2'")
    
    parser.add_argument("-style_layers", metavar="", nargs="+", type=str, default=None,
                        help="""Content layer used for content loss.
                        Default is ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']""")
    
    parser.add_argument("-content_weight", metavar="", default=0.025, type=float,
                        help="Weight of content. Default is 0.025")
    
    parser.add_argument("-style_weights", metavar="", nargs="+", default=[1.], type=float,
                        help="Weights of style, can be multiple for multiple styles. Default is 1.0")
    
    parser.add_argument("-total_variation_weight", metavar="", default=8.5e-5, type=float,
                        help="Total Variation weight. Default is 8.5e-5")

    parser.add_argument("-num_iter", default=20, metavar="", type=int,
                        help="Number of iterations. Default is 20")
    
    parser.add_argument("-model", default="vgg16", metavar="", type=str,
                        help="Choices are 'vgg16' and 'vgg19'. Default is 'vgg16'")
    
    parser.add_argument("-rescale_image", metavar="", default="False", type=str,
                        help="Rescale generated image to original image dimensions. Default is False")
    
    args = parser.parse_args()
    
    return args

def str_to_bool(s):
    """
    This function converts a string into a boolean value
    Args:
        s [str]: string representing tre value
    Returns:
        b [bool]: boolean value of string representation
        
    """
    if s.lower() in ["true", "yes", "y", "t", "1"]:
        b = True
    elif s.lower() in ["false", "no", "f", "n", "0"]:
        b = False
    else:
        print("boolean string not correctly specified")
        sys.exit(1)
    return b

def info_print(args):
    """
    This function prints the input arguments from argparse when calling this script via python shell.
    Args:
        args [argparse.Namespace]: argument namespace from main.py
    Returns:
        None
    """
    print(52*"-")
    print("Neural Style Transfer with Keras:")
    print(52*"-")
    print("Python main programm with following input arguments:")
    print(52*"-")
    for arg in vars(args):
        print (arg, ":", getattr(args, arg))
    print(52*"-")
    return None

###############################################################################
    

########################## Image Processing Functions #########################
def preprocess_image(image_path, img_shape, preprocess_input):
    """
    This function loads an image, preprocess that image and returns it as an numpy.ndarray
    The preprocessing includes resizing the image, adding a batch dimension, and centering the image based on the applied imagenet dataset.
    Args:
        image_path [str]: relative path of image
        img_shape [tuple]: tuple of desired image shape to resize 
        preprocess_input [keras.applications. ] model specific preprocess function from keras. Either VGG16 or VGG19 
    Returns:
        img [numpy.ndarray]: preprocessed image as numpy.ndarray
    """
    ## Create Image object
    img = load_img(image_path, target_size=img_shape)
    ## Parse Image object into numpy array
    img = img_to_array(img)
    ## Add "index"/"batch" axis 
    img = np.expand_dims(img, axis=0)
    ## Preprocess images in terms of zero-centering by mean pixel from ImageNet DataSet
    #  For detailed scaling have a look from line 157: 
    #  https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    img = preprocess_input(img) # 'cafe': "RGB" -> "BGR"
    return img

def deprocess_image(x, img_shape):
    """
    This function returns the mean centering operation made with preprocess_image/input when 0 centering
    the input image with the imagenet dataset mean. Hence for each channel the mean will be added again (line)
    Note this function works with both: tensorflow and theano backend.
    The image numpy array will be integer clipped onto the range 0-255.
    Args:
        x [numpy.ndarray] image to be deprocessed, dtype is float
        img_shape [tuple] shape which the image has
    Returns:
        x [numpy.ndarray] deprocessed image, dtype is integer
    """
    ## Theano
    if K.image_data_format() == "channels_first":
        x = x.reshape((3, img_shape[0], img_shape[1]))
        x = x.transpose((1, 2, 0))
    ## Tensorflow
    else:
        x = x.reshape((img_shape[0], img_shape[1], 3))
    ## Remove zero-center by mean pixel from ImageNet Dataset. Line:139
    #  https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # "BGR"->"RGB" because 'cafe' was used in preprocess_image() before
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

###############################################################################
    
################ Feature representations and loss calculations ################

def gram_matrix(x):
    """
    This function computes the gram matrix for a feature representation layer l as innerproduct.
    Args:
        x [numpy.ndarray]: Note this array is the feature map for one layer. The indexing is as follow: x.shape = (filters_l, height, width) where filters_l
                           is the number of filters in layer l. Hence each filter representation (containing filter_s matrices) are 2.dimensional arrays.
                           To compute the gram matrix the imput 3D tensor with filters_l filters
                           along the first dimension will be flattened into a 2D tensor using K.batch_flatten() and the gram matrix will be
                           computed using the gram formula: http://mathworld.wolfram.com/GramMatrix.html
    Returns:
        gram [numpy.ndarray]: Computed gram matrix where gram.shape[0] = filters_l, and gram.shape[1] = height*width at layer l.
    """
    assert K.ndim(x) == 3
    if K.image_data_format() == "channels_first":
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def content_loss(content, generated):
    """
    This function computes the content loss as defined in the paper "A Neural Algorithm of Artistic Style".
    Note the arithmetic operation "-" will be done on scalar level (elementwise on 3D tensor) In the paper the filter layers of one feature map l will
    be column concatenaded such that <content> and <generated> are 2D tensors (matrices).
    Args: 
        content [numpy.ndarray]:(3D tensor) Feature representation for content image on a specific (content) layer.
                                Shape tensorflow backend: (filters_l, height_l, width_l) at this layer.
        generated [numpy.ndarray]:(3D tensor) Feature representation for generated image on a specific (content) layer.
                                Shape tensorflow backend: (filters_l, height_l, width_l) at this layer.
    Returns:
        content_loss [float]: scalar value for content loss
    """
    assert K.ndim(content) == 3
    assert K.ndim(generated) == 3
    
    content_loss = 0.5*K.sum(K.square(generated - content))
    return content_loss

def style_loss(style, generated, img_size, session):
    """
    This function computes the style loss as defined in the paper "A Neural Algorithm of Artistic Style".
    Calls defined python function gram_matrix.
    Args:
        style [numpy.ndarray]:(3D tensor) Feature representation of style image on a specific (style) layer
        generated [numpy.ndarray]:(3D tensor) Feature representation of generated image on a specific (style) layer
        img_size [tuple] size of the generated image. Used for keras implementation of style loss factor
        session [backend.python.client.session.Session] either tensorflow or theano backend session to compute tensor shape. Used for paper implementation of style loss factor
    Returns:
        style_loss [float]: scalar value for style loss
    """
    assert K.ndim(style) == 3
    assert K.ndim(generated) == 3
    
    ## Get tensor shapes of generated feature map
    g_shape = K.shape(generated).eval(session=session)
    if K.image_data_format() == "channels_first":
        ## permute axis to get tensorflow shape representation with channel_last
        g_shape = (g_shape[1], g_shape[2], g_shape[0])
            
    S = gram_matrix(style)
    G = gram_matrix(generated)

    ## Number of filters, last channel if tensorflow. if theano, the shape was permuted such that g_shape has last channel tensorflow property
    N_l = g_shape[2]
    M_l = g_shape[0] * g_shape[1]
    
    ## keras implementation
    if True:
        channels = 3
        size = img_size[0] * img_size[1]
    else:
        ## paper implementation
        channels = N_l
        size = M_l
    
    style_loss = K.sum(K.square(S - G)) / (4.0 * (channels ** 2) * (size ** 2))
    return style_loss

def total_variation_loss(x, img_size):
    """
    This function computes the total variation loss as an additional loss to the content and style loss from the paper "A Neural Algorithm of Artistic Style".
    Args:
        x [numpy.ndarray] generated image with index axis as first dimension. 
    Returns:
        total_variation_loss [float]
    """
    assert K.ndim(x) == 4
    if K.image_data_format() == "channels_first":
        a = K.square(
            x[:, :, :img_size[0] - 1, :img_size[1] - 1] - x[:, :, 1:, :img_size[1] - 1])
        b = K.square(
            x[:, :, :img_size[0] - 1, :img_size[1] - 1] - x[:, :, :img_size[0] - 1, 1:])
    else:
        a = K.square(
            x[:, :img_size[0] - 1, :img_size[1] - 1, :] - x[:, 1:, :img_size[1] - 1, :])
        b = K.square(
            x[:, :img_size[0] - 1, :img_size[1] - 1, :] - x[:, :img_size[0] - 1, 1:, :])
        
    total_variation_loss =  K.sum(K.pow(a + b, 1.25))
    return total_variation_loss


def eval_loss_and_grads(x, img_size, f_outputs):
    """
    This function computes the loss value of a function and extract the gradients.
    Args:
        x [numpy.ndarray]: 3D tensor (generated) image
        img_size [tuple]: generated image size
        f_outputs [function] : function to output loss and gradients as list
    Returns:
        loss_value [float]: scalar loss value for (a certain generated image) x
        grad_values [numpy.ndarray] 1D array of gradient values
    """
    if K.image_data_format() == "channels_first":
        x = x.reshape((1, 3, img_size[0], img_size[1]))
    else:
        x = x.reshape((1, img_size[0], img_size[1], 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype("float64")
    else:
        grad_values = np.array(outs[1:]).flatten().astype("float64")
    return loss_value, grad_values


def main():
    
    ## retrieve arguments and print out in shell
    args = args_parser()
    ## print out information on shell
    info_print(args)
    
    ## create output directory if not available ##
    
    #### Keras Model Loading ####
    if args.model.lower() == "vgg16":
        from keras.applications.vgg16 import VGG16 as keras_model, preprocess_input 
    elif args.model.lower() == "vgg19":
        from keras.applications.vgg19 import VGG19 as keras_model, preprocess_input
    
    ## Define local variables in main environment
    if not "content/" in args.content_image_path:
        content_image_path = "content/" + args.content_image_path
        base_path = args.content_image_path
    else:
        content_image_path = args.content_image_path
        base_path = args.content_image_path[-1]
    
    ## remove file extension
    base_path = os.path.splitext(base_path)[0]
    
    output_subdir = args.output_subdir
    if output_subdir is None:
        ## Create output subdirectory
        output_subdir = "output/{}".format(base_path)
        if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
    else:
        if not "output/" in output_subdir:
            output_subdir = "output/" +  output_subdir
        if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
        
    if not "style/" in args.style_image_path:
        style_image_path = "style/" + args.style_image_path
    else:
        style_image_path = args.style_image_path
    
    init_image = args.init_image
    image_width = args.image_width
    image_height = args.image_height
    img_size = (image_height, image_width)
    content_weight = args.content_weight
    style_weights = args.style_weights
    total_variation_weight = args.total_variation_weight
    num_iter = args.num_iter
    model = args.model
    rescale_image = str_to_bool(args.rescale_image)
    content_layer = args.content_layer
    if args.style_layers == None:
        style_layers = ['block1_conv1', 'block2_conv1',
                        'block3_conv1', 'block4_conv1',
                        'block5_conv1']
    else:
        style_layers = args.style_layers
        
    print(style_layers)
    
    original_size = Image.open(content_image_path).size
    
    ###### Content Image ######
    ## Get preprocessed content image array
    content_image = preprocess_image(content_image_path, img_size, preprocess_input)
    ## Parse content_image numpy array as Keras Backend Variable
    content_image = K.variable(content_image, dtype="float32", name="content_image") 
    
    ###### Style Image ######
    ## Get preprocessed style image array
    style_image = preprocess_image(style_image_path, img_size, preprocess_input)
    ## Parse style image numpy array as Keras Backend Variable
    style_image = K.variable(style_image, dtype="float32", name="style_image")
    
    ###### Generated Image ######
    ## Init generated image as numpy array and parse into Keras Backend Variable
    if init_image == "content":
        generated_image = preprocess_image(content_image_path, img_size, preprocess_input)
    elif init_image == "random":
        generated_image = np.random.randint(256, size=(image_width, image_height, 3)).astype("float64")
        generated_image = preprocess_input(np.expand_dims(generated_image, axis=0))
    else:
        import sys
        print("wrong init_image")
        sys.exit(1)
    fname = output_subdir + "/generated_image_at_iteration_0.jpg"
    save_img(path=fname, x=generated_image[0])
    
    ## Define generate image variable placeholder for later optimization
    # Theano
    if K.image_data_format() == "channels_first":
        generated_image_placeholder = K.placeholder(shape=(1, 3, image_height, image_width))
    # Tensorflow
    else:
        generated_image_placeholder = K.placeholder(shape=(1, image_height, image_width, 3))
        
    
    ###### Initialize one keras models with one input tensors which is concatenated by 3 images ######
    input_tensor = K.concatenate([content_image,
                                  style_image,
                                  generated_image_placeholder], axis=0)
    ## input_tensor is a 4D tensor, with shape (3, image_height, image_width, 3) where the first 3 is the concatenation of 3 images and last 3 the color channel (tf)
    
    # build the keras network with our 3 images as input
    model = keras_model(input_tensor=input_tensor, weights='imagenet', include_top=False)


    # get the symbolic outputs of each layer (we gave them unique names). [Feature representations/maps in form of 4D tensors at each layer]
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    
    # combine these loss functions into a single scalar
    loss = K.variable(0.0)
    layer_features = outputs_dict[content_layer]
    
    ############# Content extraction: #############
    # retrieve content_image output for content_layer 
    content_image_features = layer_features[0, :, :, :]
    # retrieve generated_image output from content_layer
    generated_image_features = layer_features[2, :, :, :]
    # get loss containing only content loss
    loss = loss +  content_weight * content_loss(content_image_features,
                                          generated_image_features)
    
    ############# Style Extraction:  #############
    if len(style_weights) == 1:
        style_weights = [style_weights[0]] * len(style_layers)
    else:
        assert len(style_weights) == len(style_layers)
        style_weights = [float(style_weight) for style_weight in style_weights]
    
    session = K.get_session()    
    for style_weight, layer_name in zip(style_weights,style_layers):
        ## get feature activations from layers
        layer_features = outputs_dict[layer_name]
        ## retrieve style_image output activations for a style_layer
        style_image_features = layer_features[1, :, :, :]
        ## retrieve generated_image output activations for a style_layer
        generated_image_features = layer_features[2, :, :, :]
        ## get loss containing content loss and style loss
        loss = loss + (style_weight / len(style_layers)) * style_loss(style_image_features, generated_image_features,
                      img_size, session)
        
    ## get loss containing content loss, style loss and total variation loss
    loss = loss + total_variation_weight * total_variation_loss(generated_image_placeholder, img_size)
    
    # get the gradients of the generated image wrt. the loss
    grads = K.gradients(loss, generated_image_placeholder)
    
    # Define outputs list to have loss included
    outputs = [loss]
    
    # add the gradients to the outputs instance
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)
        
    ## Define keras function with input the placeholder of the generated image and output the {loss and gradients} for learning
    f_outputs = K.function(inputs=[generated_image_placeholder], outputs=outputs)
    
    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None
    
        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x, img_size, f_outputs)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value
    
        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values
    
    # this Evaluator class makes it possible
    # to compute loss and gradients in one pass
    # while retrieving them via two separate functions,
    # "loss" and "grads". This is done because scipy.optimize
    # requires separate functions for loss and gradients,
    # but computing them separately would be inefficient.
    
    evaluator = Evaluator()
    
    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss
    loss_history = [None] * num_iter
    for i in range(num_iter):
        print("Start of iteration:", i+1)
        start_time = time.time()
        generated_image, loss_history[i], info = fmin_l_bfgs_b(evaluator.loss, generated_image.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print("Current loss value:", loss_history[i])
        # save current generated image
        img = deprocess_image(generated_image.copy(), img_shape=img_size)
        if rescale_image:
            img = array_to_img(img[0])
            img = img.resize(original_size)
            img = img_to_array(img)
            
        fname = output_subdir + "/generated_image_at_iteration_%s.png" % str(i+1)
        save_img(path=fname, x=img)
        end_time = time.time()
        print("Image saved at:", fname)
        print("Iteration %s completed in %ds" % (str(i+1), end_time - start_time))
        
    # summarize history for loss
    plt.figure(3,figsize=(7,5))
    plt.plot(loss_history)
    plt.title("loss process during neural style transfer")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.savefig(output_subdir + "/loss_history.jpg")
    plt.close()
    

if __name__ == "__main__":
    main()