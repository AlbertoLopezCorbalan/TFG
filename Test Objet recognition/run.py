import tensorflow as tf
import mobilenet_v2
import imagenet

# Colocar la imagen en el mismo directorio que run.py y modifiar el nombre aquí para realizar pruebas.
IMAGEN_ANALIZAR = 'Switch.jpg'

tf.reset_default_graph()

file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
  logits, endpoints = mobilenet_v2.mobilenet(images)
  
ema = tf.train.ExponentialMovingAverage(0.999)
vars = ema.variables_to_restore()

saver = tf.train.Saver(vars)

with tf.Session() as sess:
  saver.restore(sess,  'mobilenet_v2_1.0_224.ckpt')
  x = endpoints['Predictions'].eval(feed_dict={file_input: IMAGEN_ANALIZAR})
label_map = imagenet.create_readable_names_for_imagenet_labels()  
print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())

