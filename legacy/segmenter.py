from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import TensorflowUtils as utils
import test_image_reader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("model_dir", "vgg_model/", "Path to vgg model mat")
tf.flags.DEFINE_bool("debug", "True", "Debug mode: True/ False")
tf.flags.DEFINE_integer("NUM_OF_CLASSESS", "2", "Path to vgg model mat")
tf.flags.DEFINE_integer("IMAGE_SIZE", "800", "Iamge size")
tf.flags.DEFINE_string("model_path", "trained_models/", "Path to trained models")
tf.flags.DEFINE_string("input_path", "input/", "Path to input")
tf.flags.DEFINE_string("output_path", "output/", "Path to output")

MODEL_URL = 'imagenet-vgg-verydeep-19.mat'

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4')

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    return net

def inference(image, keep_prob):
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, FLAGS.NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([FLAGS.NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, FLAGS.NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], FLAGS.NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, FLAGS.NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([FLAGS.NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]), name="entropy")))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    sess = tf.Session()

    print("Restoring Model...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.model_path, sess.graph)
    saver.restore(sess, os.path.join(FLAGS.model_path, "model.ckpt"))

    print('Testing...')
    clear_folder(FLAGS.output_path)
    valid_images, names = test_image_reader._read_images_fcn(FLAGS.input_path, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE)
    for i in range(len(valid_images)):
        name = str(names[i]).replace('.jpg','')
        print('testing iamge: ' + name)
        im = np.zeros([1, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 3], dtype = np.uint8)
        label = np.zeros([1, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1], dtype = np.uint8)
        im[0] = valid_images[i]
        pred = sess.run(pred_annotation, feed_dict={image: im, annotation: label, keep_probability: 1.0})
        # output black image
        pred = np.squeeze(pred, axis=3)
        utils.save_image(pred[0].astype(np.uint8), FLAGS.output_path, name=name+"_pred")
    utils.remove_test_log(FLAGS.model_path)
    return 'done'
	
def clear_folder(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.mkdir(path)

if __name__ == "__main__":
    tf.app.run()
	
# python segmenter.py --model_path trained_models/logs_pole/ --input_path output/ori/ --output_path output/res/