import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_saturation(image, lower=0.5, upper=1)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1)
        image = tf.image.random_hue(image, max_delta=0.3)
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox_begin, bbox_size, _ =tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox, min_object_covered=0.1)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = tf.image.resize_images(distorted_image,  size=[height, width], method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return  distorted_image

image_raw_data = tf.gfile.FastGFile("/tmp/data/image/m.jpg", "rb").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7],[0.35, 0.47, 0.5, 0.56]]])

    for i in range(6):
        result = preprocess_for_train(img_data, 240, 221, boxes)
        plt.imshow(result.eval())
        plt.show()
