import tensorflow as tf

class Layer:
 #========================================= Layer Ops  ==========================================#
    def lrelu  (self, x, leak=0.25, name="lrelu"):
        return tf.maximum(x, leak*x)
    def conv   (self, x, k_h=3 ,k_w=3 ,k_stride=1,output_dim=64,if_batch=True,name='conv'):
        # x : input feature map
        # k_h,k_w: kernel height, width
        # output_dimension
        # name

        with tf.variable_scope(name):
            weight = tf.get_variable('weight',
                                      [k_h,k_w,x.get_shape()[-1], output_dim],
                                      initializer=tf.random_normal_initializer(stddev=0.02))

            bias   = tf.get_variable('bias',
                                      [output_dim],
                                      initializer=tf.constant_initializer(0.0))

            lay_1  = tf.nn.conv2d(x, weight, strides=[1,k_stride,k_stride,1], padding='VALID')+bias

            if if_batch:
                lay_2  = tf.contrib.layers.batch_norm(inputs=lay_1)
                return lay_2
            else:
                return lay_1
    def up_conv(self, x, k_h, k_w,k_stride,out_ch_size,output_dim=64,if_batch=True,batch_size=4,padding='VALID', name='up_conv'):

        with tf.variable_scope(name):

            weight = tf.get_variable('weight',
                                     [k_h,k_w, output_dim,x.get_shape()[-1]],
                                     initializer=tf.random_normal_initializer(stddev=0.02))
            bias   = tf.get_variable('bias',
                                     [output_dim],
                                     initializer=tf.constant_initializer(0.0))

            lay_1  = tf.nn.conv2d_transpose(x, weight,
                                            output_shape=tf.constant([batch_size,out_ch_size,out_ch_size,output_dim]),
                                            strides=[1,k_stride,k_stride,1],
                                            padding='VALID')+ bias
            if if_batch:
                lay_2  = tf.contrib.layers.batch_norm(inputs=lay_1)
                return lay_2
            else:
                return lay_1
    def linear (self, input_, output_size, if_batch=True , name='linear'):
        shape = input_.get_shape().as_list()
        #print('Fully-Connected Input Size:')
        #print(shape[1])

        with tf.variable_scope(name):
            weight = tf.get_variable("weight",
                                     [shape[1], output_size],
                                     tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.02))

            bias   = tf.get_variable("bias",
                                     [output_size],
                                     initializer=tf.constant_initializer(0.0))

            output = tf.matmul(input_, weight) + bias

            if if_batch:
                lay_2  = tf.contrib.layers.batch_norm(inputs=output)
                return lay_2
            else:
                return output
