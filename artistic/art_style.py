#--------------------------------------------------------------------------------
# Neural Style transfer Algorithm based on
#   Leon A Gatys. "A neural algorithm of artistic.pdf"
#   Andrew NG, coursera Deep learning: CNN 
##
# Implemented By GongPing @2017-11-27
#
# Note, this algorithm depends on transfer learning from VGG-19, 
# Detailed information is shown in nst_utils.py
#--------------------------------------------------------------------------------

import os
import sys
import argparse
import matplotlib.pyplot as plt
import PIL
import matplotlib.image as mpimg
import nst_utils

import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

class ArtStyle(object):
    """
    Generate a new image based on a style image and a content image.
    Based on the following paper: 
        Leon A Gatys. "A neural algorithm of artistic.pdf"
        Andrew NG, coursera Deep learning: CNN
    """
    def __init__(self):
        self.model = nst_utils.load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
        self.STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]
        
        self.input_shape = (300, 400)  # (hight, width) of VGG16 input layer
        self.content_image = None
        self.style_image = None

    def print_vgg_net(self):
        for key, val in self.model.items():
            print("%-20s\t%s" % (key, val))
        
    def resize_input(self, fname):
        im = PIL.Image.open(fname)
        width, height = im.size
        name, ext = fname.rsplit('.', maxsplit=1)
        print("shape of original image: (%d, %d)" % (height, width))
        if width < height:
            im = im.resize(self.input_shape) # PIL (w, h)
            name_resized = name + '_h400w300' + '.' + ext
        else:
            im = im.resize(tuple(reversed(self.input_shape))) # PIL (w, h)
            name_resized = name + '_h300w400' + '.' + ext
        width, height = im.size
        print("shape of resized image: (%d, %d)" % (height, width))
        im.save(name_resized)
        return name_resized

    def read_check_image(self, fname):
        plt.subplot(1, 2, 1)
        image = mpimg.imread(fname)
        plt.imshow(image)
        # prepare for VGG input
        plt.subplot(1, 2, 2)
        h, w, c = image.shape
        if h > w:
            flip = True
            image = np.transpose(image, axes=(1, 0, 2))
        else:
            flip = False
        image = nst_utils.reshape_and_normalize_image(image)
        plt.imshow(image[0])
        print("shape of normalized image:", fname, image.shape)
        plt.show()
        return image, flip
    
    def load_input(self, fname):
        name_resized = self.resize_input(fname)
        return self.read_check_image(name_resized)

    def load_content(self, fname):
        self.content_image, self.content_flip = self.load_input(fname)

    def load_init(self, fname):
        if fname is None:
            self.init_image = nst_utils.generate_noise_image(self.content_image)
        else:
            self.init_image, self.init_flip = self.read_check_image(fname)
        print("shape of initial image: ", self.init_image.shape)

    def content_cost(self):
        """ 
        Pre Calculate activation of Content Image @ conv4_2
        and return the tensor for content cost
        """
        sess.run(self.model['input'].assign(self.content_image))
        out = self.model['conv4_2']
        a_C = sess.run(out)
        # Build the content cost computation graph
        a_G = out
        return self.compute_content_cost(a_C, a_G)

    def compute_content_cost(self, a_C, a_G):
        """
        Computes the content cost
        
        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
        Returns: 
        J_content -- scalar that you compute using equation 1 above.
        """
    
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
        # Reshape a_C and a_G (≈2 lines)
        a_C_unrolled = tf.reshape(tf.transpose(a_C, perm=[0, 3, 1, 2], name='transpose_C'), [m, n_C, n_H * n_W])
        a_G_unrolled = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2], name='transpose_G'), [m, n_C, n_H * n_W])
        
        # compute the cost with tensorflow (≈1 line)
        J_content = tf.scalar_mul(1/4, tf.reduce_mean(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))))
    
        return J_content
    
    def gram_matrix(self, A):
        """
        Return the GRAM matrix for style cost 

        Argument:
            shape (n_C, n_H*n_W)        
        Returns:
            GA -- Gram matrix of A, of shape (n_C, n_C)
        """    
        GA = tf.matmul(A, A, transpose_b=True)        
        return GA

    def style_cost(self):
        """
        Pre Calculate the activation of style image's GRAM matrixes
        and return the tensor for style cost
        """
        sess.run(self.model['input'].assign(self.style_image))
        return self.compute_style_cost()

    def load_style(self, fname):
        self.style_image, self.style_flip = self.load_input(fname)    

    def compute_layer_style_cost(self, a_S, a_G):
        """
        Return Style cost tensor for one layer

        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), 
               hidden layer activations representing style of the image S 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), 
               hidden layer activations representing style of the image G
        Returns: 
        J_style_layer -- tensor representing a scalar value, 
        """
    
        # Retrieve dimensions from a_G (≈1 line)
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
   
        # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
        a_S = tf.reshape(tf.transpose(a_S, perm=[0, 3, 1, 2]), shape=[m, n_C, n_H*n_W])
        a_G = tf.reshape(tf.transpose(a_G, perm=[0, 3, 1, 2]), shape=[m, n_C, n_H*n_W])

        # Computing gram_matrices for both images S and G (≈2 lines)
        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)

        # Computing the loss (≈1 line)
        J_style_layer = tf.divide(tf.reduce_sum(tf.square(tf.subtract(GS, GG))),
                                  (2*n_H*n_W*n_C)**2)
        return J_style_layer


    def compute_style_cost(self):
        """
        Computes and return the tensor of overall style cost from several chosen layers
        """    
        # initialize the overall style cost
        J_style = 0
        for layer_name, coeff in self.STYLE_LAYERS:
            # Select the output tensor of the currently selected layer
            out = self.model[layer_name]
            # Pre calculate style layer activation from the current layer
            a_S = sess.run(out)
            # Set a_G to be the hidden layer activation from same layer
            a_G = out  # Do not run yet, wait until assign 'generated_image' to input        
            # Compute style_cost for the current layer
            J_style_layer = self.compute_layer_style_cost(a_S, a_G)
            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += coeff * J_style_layer # Remember this is a tensor from multiple layers
        return J_style
    
    def total_cost(self, J_content, J_style):
        # self.J_content = self.content_cost()
        # self.J_style = self.style_cost()
        alpha = 10 # 10, hyperparameters from Andrew NG
        beta = 10  # 40, hyperparameters from Andrew NG
        return tf.add(alpha * J_content, beta * J_style)


    def fit(self, learning_rate=2.0, num_iterations=200):
        history = []
        # Assign the content image to be the input of the VGG model.
        # sess.run(self.model['input'].assign(self.content_image))
        # out = self.model['conv4_2']
        # a_C = sess.run(out)
        # a_G = out
        # J_content = self.compute_content_cost(a_C, a_G)
        
        J_content = self.content_cost()
        J_style = self.style_cost()
        J = self.total_cost(J_content, J_style)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(J)

        # init global variables
        sess.run(tf.global_variables_initializer())
        # run the noisy input image, use assign()
        sess.run(self.model['input'].assign(self.init_image))
        if not os.path.isdir('output'):
            os.mkdir('output')
        for i in range(num_iterations):
            sess.run(self.train_step)
            generated_image = sess.run(self.model['input'])
            # Print every 20 iteration.
            if i%20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                # save current generated image in the "/output" directory
                nst_utils.save_image("output/" + str(i) + ".png", generated_image)
                history.append((Jt, Jc, Js))
        # save last generated image
        nst_utils.save_image('output/generated_image.jpg', generated_image)
        
        return history
    
    def test_content_cost(self):
        tf.reset_default_graph()
        
        with tf.Session() as test:
            tf.set_random_seed(1)
            a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
            a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
            J_content = self.compute_content_cost(a_C, a_G)
            print("J_content = " + str(J_content.eval()))

    def test_style_cost(self):
        tf.reset_default_graph()
        
        with tf.Session() as test:
            tf.set_random_seed(1)
            a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
            a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
            J_style_layer = self.compute_layer_style_cost(a_S, a_G)
    
            print("J_style_layer = " + str(J_style_layer.eval()))
        
    def test_total_cost(self):
        tf.reset_default_graph()

        with tf.Session() as test:
            np.random.seed(3)
            J_content = np.random.randn()    
            J_style = np.random.randn()
            J = self.total_cost(J_content, J_style)
            print("J = " + str(test.run(J)))
        
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Neruo Style Transfer.')
    parser.add_argument('-c', '--content', nargs=1, required=True, help='the style image')
    parser.add_argument('-s', '--style', nargs=1, required=True, help='the style image')
    parser.add_argument('-i', '--init', nargs='?', type=int, help='the initial image')
    parser.add_argument('--num_iter', nargs='?', default=300, help='the initial image')

    args = parser.parse_args()
    print(args)
    model = ArtStyle()
    model.load_content(args.content[0])
    model.load_style(args.style[0])
    model.load_init(args.init)
    if args.num_iter == 0:
        sys.exit(1)
    #- model.test_content_cost()
    #- model.test_style_cost()
    #- model.test_total_cost()
    
    history = model.fit(learning_rate=2.0, num_iterations=int(args.num_iter))
    plt.plot(history)
    plt.show()

    import pickle
    pickle.dump(history, open("output/history.p", "wb"))
    print("---------------------------------")
    print("Art Style successfully finished! ")
    print("---------------------------------")
