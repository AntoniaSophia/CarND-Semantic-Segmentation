from graph_utils import load_graph
import tensorflow as tf
from tensorflow.python.platform import gfile
import helper

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

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

        #saver = tf.train.import_meta_graph('./runs/semantic_segmentation_model.ckpt.meta')
        #saver.restore(sess,tf.train.latest_checkpoint('./runs/semantic_segmentation_model.ckpt'))

        #sess, _ = load_graph('./runs/frozen_graph.pb')
        sess, _ = load_graph('./runs/optimized_graph.pb')
        graph = sess.graph
        adam_angst = graph.get_tensor_by_name('adam_logit:0')
        image_input = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        print('hallo')
        print('Adam Angst = ',adam_angst)
        logits = graph.get_tensor_by_name('adam_logit:0')
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())


        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)
    
        print("%d ops in the final graph." % len(output_graph_def.node))


        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

