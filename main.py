import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from graph_utils import load_graph

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, layer3, layer4, layer7


tests.test_load_vgg(load_vgg, tf)


def conv_1x1(x, num_outputs):
    kernel_size = 1
    stride = 1
    return tf.layers.conv2d(x, num_outputs, kernel_size, stride,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                            kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

def upsample(x, num_outputs, kernel_size,stride):
    return tf.layers.conv2d_transpose(x, num_outputs,kernel_size ,stride, padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))    

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Take pretrained VGG as basic network
    
    #####################################
    # Encoder part
    #####################################

    # 1x1 convolution of vgg layer7, layer4 and layer3

    # Output dimension 5x18x2
    layer7_conv1x1 = conv_1x1(vgg_layer7_out, num_classes)

    # Output dimension 10x36x2
    layer4_scaled = tf.multiply(vgg_layer4_out, 0.01) #Scaling according to: https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
    layer4_conv1x1 = conv_1x1(layer4_scaled, num_classes)

    # Output dimension 20x72x2
    layer3_scaled = tf.multiply(vgg_layer3_out, 0.0001) #Scaling according to: https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
    layer3_conv1x1 = conv_1x1(layer3_scaled, num_classes)

    #####################################
    # Decoder part
    #####################################
    # upsample by factor 2 --> 10x36x2
    upsample_1 = upsample(layer7_conv1x1, num_classes,4,(2,2))
    # add skip connection to layer4         
    skip_1 = tf.add(upsample_1, layer4_conv1x1)

    # upsample again by factor 2 --> 20x72x2
    upsample_2 = upsample(skip_1, num_classes,4,(2,2))
    # add skip connection to layer4         
    skip_2 = tf.add(upsample_2, layer3_conv1x1)


    # upsample again by factor 8 --> 160x576x2
    nn_last_layer = upsample(skip_2, num_classes,16,(8,8))

    return nn_last_layer

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # Reshape predictions and labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="adam_logit")
    correct_labels = tf.reshape(correct_label, (-1,num_classes))

    # Define loss function - note: the regularizer must be considered here!
    # see e.g. https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) # This is a list of the individual loss values, so we still need to sum them up.
    reg_constant = 0.01  # Choose an appropriate one.
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_labels))
    total_loss = tf.add(cross_entropy_loss,  reg_constant * sum(regularization_losses), name='total_loss') # Using total loss according to: https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)

    # Define training operation (otimize = minimize loss)
    train_op = optimizer.minimize(total_loss)

    return logits, train_op, total_loss
tests.test_optimize(optimize)




def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param image_input: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    iteration = 0
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            iteration=iteration+1
            _, loss = sess.run([train_op, cross_entropy_loss],
                            feed_dict={image_input: image,
                                        correct_label: label,
                                        keep_prob: 0.8,
                                        learning_rate: 1e-4})
            print("Epoch: {}, batch: {}, loss: {}".format(epoch+1, iteration, loss))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Create a TensorFlow configuration object. This will be 
    # passed as an argument to the session.
    config = tf.ConfigProto()
    # JIT level, this can be set to ON_1 or ON_2 
    jit_level = tf.OptimizerOptions.ON_2
    config.graph_options.optimizer_options.global_jit_level = jit_level
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    #with tf.Session(config=config) as sess:
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 50
        batch_size = 8
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        output_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        correct_label = tf.placeholder(tf.int32, shape=[None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())



        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, 
                 image_input,correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

   
        # save the (trained) model 
        # We use a built-in TF helper to export variables to constants
        output_node_names = 'adam_logit'
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 
        saver = tf.train.Saver()
        saver.save(sess, './runs/semantic_segmentation_model.ckpt')
        tf.train.write_graph(tf.get_default_graph().as_graph_def(), '', './runs/base_graph.pb', False)
        tf.train.write_graph(output_graph_def, '', './runs/frozen_graph.pb', False)
        print("%d ops in the final graph." % len(output_graph_def.node))


        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
