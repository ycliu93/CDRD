import tensorflow as tf
import layer


class Nets:
    def __init__(self, input_z_size,emb_size, class_num, code_dim, img_size):
        lay = layer.Layer()
        self.linear  = lay.linear
        self.up_conv = lay.up_conv
        self.conv    = lay.conv
        self.lrelu   = lay.lrelu
        self.input_z_size = input_z_size
        self.emb_size     = emb_size
        self.class_num    = class_num
        self.code_dim     = code_dim
        self.img_size     = img_size

    def generator(self, input_img,  ushared_name,shared_reuse=False,unshared_reuse=False):
        with tf.variable_scope('Generator'):
            with tf.variable_scope('Shared'):
                if shared_reuse:
                    tf.get_variable_scope().reuse_variables()
                l1         = tf.reshape(input_img, [input_img.get_shape().as_list()[0], 1 , 1, self.input_z_size+self.code_dim])
                l2         = self.lrelu(self.up_conv(l1        , k_h=2,k_w=2,k_stride=2, out_ch_size=2               , output_dim=1024 ,if_batch = False, batch_size=input_img.get_shape().as_list()[0],padding='SAME', name='g_conv2'), name='g_plr_2')
                l3         = self.lrelu(self.up_conv(l2        , k_h=3,k_w=3,k_stride=2, out_ch_size=5               , output_dim=512  ,if_batch = False, batch_size=input_img.get_shape().as_list()[0],padding='SAME', name='g_conv3'), name='g_plr_3')
                l4         = self.lrelu(self.up_conv(l3        , k_h=3,k_w=3,k_stride=2, out_ch_size=12              , output_dim=256  ,if_batch = False, batch_size=input_img.get_shape().as_list()[0],padding='SAME', name='g_conv4'), name='g_plr_4')
                l5         = self.lrelu(self.up_conv(l4        , k_h=3,k_w=3,k_stride=2, out_ch_size=25              , output_dim=128  ,if_batch = False, batch_size=input_img.get_shape().as_list()[0], name='g_conv7'), name='g_plr_7')

            with tf.variable_scope(ushared_name):
                if unshared_reuse:
                    tf.get_variable_scope().reuse_variables()
                l6         = tf.tanh   (self.up_conv(l5        , k_h=4,k_w=4,k_stride=1, out_ch_size=self.img_size   , output_dim=1    ,if_batch = False, batch_size=input_img.get_shape().as_list()[0], name='g_conv8'), name='g_tanh_8')
            return l6

    def discriminator(self,input_img, ushared_name,shared_reuse=False,unshared_reuse=False):
        with tf.variable_scope('Discriminator'):
            with tf.variable_scope(ushared_name):
                if unshared_reuse:
                    tf.get_variable_scope().reuse_variables()
                l1      = self.lrelu (   self.conv(input_img , k_h=5,k_w=5,k_stride=1  , output_dim=20       , if_batch=False  , name='d_conv1' ), name='d_llr_1')
            with tf.variable_scope('Shared'):
                if shared_reuse:
                    tf.get_variable_scope().reuse_variables()

                l1_pool = tf.nn.max_pool(l1, [1,2,2,1],[1,2,2,1],padding='VALID')
                l2      = self.conv(l1_pool        , k_h=5,k_w=5,k_stride=1  , output_dim=50       , if_batch=False  , name='d_conv2' )
                l2_pool = tf.nn.max_pool(l2, [1,2,2,1],[1,2,2,1],padding='VALID')
                l3      = self.lrelu (   self.conv(l2_pool    , k_h=4,k_w=4,k_stride=1  , output_dim=500      , if_batch=False  , name='d_conv3' ), name='d_llr_3')
                l4_fl   = tf.reshape (l3, [input_img.get_shape().as_list()[0], -1 ])
                l5_fl   =                self.linear(l4_fl   , output_size = self.emb_size                    , if_batch=False  , name='d_fc1')

                # GAN : predict real or fake
                gan_out =                self.linear(l5_fl   , output_size = 2                                , if_batch=False  , name='d_fc2_RF')

                # attribute classifier
                cls_out=                self.linear(l5_fl   , output_size = self.class_num                    , if_batch=False  , name='d_fc2_PN')
            gan_out_softmax   = tf.nn.softmax(gan_out)
            cls_out_softmax   = tf.nn.softmax(cls_out)

            return gan_out_softmax,gan_out,l2, cls_out_softmax, cls_out, l5_fl
