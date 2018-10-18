import tensorflow as tf
import numpy as np
import scipy.misc
import nets
import loss
from utils import ops

class Model:
    def __init__(self,sess,summaries_dir, batch_size, input_z_size,emb_size, class_num, code_dim, img_size):
        self.sess = sess
        # Function
        net = nets.Nets(input_z_size,emb_size, class_num, code_dim, img_size)
        self.generator     = net.generator
        self.discriminator = net.discriminator

        los = loss.Loss(batch_size)
        self.class_loss    = los.class_loss
        self.class_loss2   = los.class_loss2
        self.vae_loss      = los.vae_loss
        self.percept_loss  = los.percept_loss
        self.tv_loss       = los.tv_loss

        self.opers = ops.Ops()

        self.batch_size    = batch_size
        self.input_z_size  = input_z_size
        self.class_num     = class_num
        self.code_dim      = code_dim
        self.img_size      = img_size
        self.emb_size      = emb_size
        self.gen_gan_weight       = 1
        self.gen_dis_weight       = 1
        self.dis_gan_weight       = 1
        self.dis_dis_weight       = 1
        self.summaries_dir        = summaries_dir
        self.build_model()
 #=========================================  Build   ==========================================#
    def build_model(self):
        with tf.name_scope('Input'):
            self.z         = tf.placeholder(tf.float32, shape = (self.batch_size,  self.input_z_size))
            self.z_lb      = tf.placeholder("float32", shape = (self.batch_size,  self.class_num))

            with tf.name_scope('Domain1'):
                self.X1        = tf.placeholder(tf.float32, shape = (self.batch_size,  self.img_size,  self.img_size, 1))
                self.X1_lb     = tf.placeholder("float32", shape = (self.batch_size,  self.class_num))
            with tf.name_scope('Domain2'):
                self.X2        = tf.placeholder(tf.float32, shape = (self.batch_size,  self.img_size,  self.img_size, 1))
                ## Only for test the accuracy, do not use during training stage
                self.X2_lb     = tf.placeholder("float32", shape = (self.batch_size,  self.class_num))
        with tf.name_scope('Procedure'):
            with tf.name_scope('Disentangle_coding'):
                self.noise_with_code = self.opers.disentangle_coding(self.z, self.z_lb)

            with tf.name_scope('Generator'):
                with tf.name_scope('Domain_1'):
                    self.G1_z        =  self.generator(self.noise_with_code        , 'dm1',shared_reuse=False  ,unshared_reuse=False)
                with tf.name_scope('Domain_2'):
                    self.G2_z        =  self.generator(self.noise_with_code        , 'dm2',shared_reuse=True ,unshared_reuse=False)

            with tf.name_scope('Discriminator'):
                with tf.name_scope('Domain_1'):
                    with tf.name_scope('Real'):
                        self.D_X1_gan_log, self.D_X1_gan, self.D_X1_l2 ,self.D_X1_cls_log,self.D_X1_cls, self.X1_emb  = self.discriminator(self.X1        ,'dm1',shared_reuse=False ,unshared_reuse=False)
                    with tf.name_scope('Fake'):
                        self.D_Z1_gan_log, self.D_Z1_gan, _           ,self.D_Z1_cls_log, self.D_Z1_cls,_             = self.discriminator(self.G1_z      ,'dm1',shared_reuse=True ,unshared_reuse=True)

                with tf.name_scope('Domain_2'):
                    with tf.name_scope('Real'):
                        self.D_X2_gan_log, self.D_X2_gan,self.D_X2_l2  ,self.D_X2_cls_log,self.D_X2_cls, self.X2_emb  = self.discriminator(self.X2        ,'dm2',shared_reuse=True ,unshared_reuse=False)

                    with tf.name_scope('Fake'):
                        self.D_Z2_gan_log, self.D_Z2_gan,self.D_Z2_l2  ,self.D_Z2_cls_log,self.D_Z2_cls, _            = self.discriminator(self.G2_z      ,'dm2',shared_reuse=True ,unshared_reuse=True)

                with tf.name_scope('UDA'):
                    with tf.name_scope('Classification'):
                        self.D1_correct_prediction = tf.equal(tf.argmax(self.D_X1_cls, 1), tf.argmax(self.X1_lb, 1))
                        self.D1_accuracy           = tf.reduce_mean(tf.cast(self.D1_correct_prediction, tf.float32))
                        self.D1_accuracy_sum       = tf.summary.scalar('D1_accuracy', self.D1_accuracy)

                        self.D2_correct_prediction = tf.equal(tf.argmax(self.D_X2_cls, 1), tf.argmax(self.X2_lb, 1))
                        self.D2_accuracy           = tf.reduce_mean(tf.cast(self.D2_correct_prediction, tf.float32))
                        self.D2_accuracy_sum       = tf.summary.scalar('D2_accuracy', self.D2_accuracy)

                with tf.name_scope('Discriminator_Ability'):
                    show_D_X1_r_log  ,_ = tf.split(self.D_X1_gan_log  ,[1,1],1)
                    show_D_X2_r_log  ,_ = tf.split(self.D_X2_gan_log  ,[1,1],1)

                    show_D_z1_f_log  ,_ = tf.split(self.D_Z1_gan_log  ,[1,1],1)
                    show_D_z2_f_log  ,_ = tf.split(self.D_Z2_gan_log  ,[1,1],1)

                    self.D1_real_show              = tf.reduce_mean(show_D_X1_r_log)
                    self.D2_real_show              = tf.reduce_mean(show_D_X2_r_log)
                    self.D1_fake_show              = tf.reduce_mean(show_D_z1_f_log)
                    self.D2_fake_show              = tf.reduce_mean(show_D_z2_f_log)

                    self.D1_real_sum               = tf.summary.scalar('D1_real_predict', self.D1_real_show)
                    self.D1_fake_sum               = tf.summary.scalar('D1_fake_predict', self.D1_fake_show)
                    self.D2_real_sum               = tf.summary.scalar('D2_real_predict', self.D2_real_show)
                    self.D2_fake_sum               = tf.summary.scalar('D2_fake_predict', self.D2_fake_show)

        with tf.name_scope('Loss_Function'):
            with tf.name_scope('Dis_loss'):
                with tf.name_scope('Domain_1'):
                    with tf.name_scope('Real'):
                        # gan
                        self.D_X1_gan_loss   = self.class_loss(self.D_X1_gan_log, 'pos')
                        # dis
                        self.D_X1_dis_loss   = self.class_loss2(self.D_X1_cls_log, self.X1_lb)
                    with tf.name_scope('Fake'):
                        # gan
                        self.D_z1_gan_loss   = self.class_loss(self.D_Z1_gan_log, 'neg')
                        # dis
                        self.D_z1_dis_loss   = self.class_loss2(self.D_Z1_cls_log, self.z_lb)

                    self.D_X1_real_loss    = self.D_X1_gan_loss
                    self.D_X1_real_loss2   = self.D_X1_dis_loss

                    self.D_z1_fake_loss    = self.D_z1_gan_loss
                    self.D_z1_fake_loss2   = self.D_z1_dis_loss

                    self.D_X1_loss  = (self.D_X1_real_loss +  1* self.D_z1_fake_loss)/2
                    self.D_X1_loss2 = (self.D_X1_real_loss2 + 1* self.D_z1_fake_loss2)/2

                    self.D_z1_fake_loss2_sum  = tf.summary.scalar('D_z1_fake_loss2', self.D_z1_fake_loss2)
                    self.D_X1_real_loss2_sum  = tf.summary.scalar('D_X1_real_loss2', self.D_X1_real_loss2)

                with tf.name_scope('Domain_2'):
                    with tf.name_scope('Real'):
                        # gan
                        self.D_X2_gan_loss   = self.class_loss(self.D_X2_gan_log, 'pos')
                        ### NO LABEL IN TARGET DOMAIN
                    with tf.name_scope('Fake'):
                        # gan
                        self.D_z2_gan_loss   = self.class_loss(self.D_Z2_gan_log, 'neg')
                        # dis
                        self.D_z2_dis_loss   = self.class_loss2(self.D_Z2_cls_log, self.z_lb)

                    self.D_X2_real_loss      = self.D_X2_gan_loss

                    self.D_z2_fake_loss      = self.D_z2_gan_loss
                    self.D_z2_fake_loss2     = self.D_z2_dis_loss

                    self.D_X2_loss  = (self.D_X2_real_loss +  1* self.D_z2_fake_loss)/2
                    self.D_X2_loss2 = (1* self.D_z2_fake_loss2)/2

                    self.D_z2_fake_loss2_sum  = tf.summary.scalar('D_z2_fake_loss2', self.D_z2_fake_loss2)

                self.D_loss      = self.dis_gan_weight * (self.D_X1_loss+self.D_X2_loss)   # RF loss
                self.D_loss_sum  = tf.summary.scalar('D_gan_loss', self.D_loss)
                self.D_loss2     = self.dis_dis_weight * (self.D_X1_loss2+self.D_X2_loss2) # PN Classification loss
                self.D_loss2_sum = tf.summary.scalar('D_dis_loss', self.D_loss2)


            with tf.name_scope('Gen_loss'):
                with tf.name_scope('Domain_1'):
                    # gan
                    self.G_z1_gan_loss   = self.class_loss(self.D_Z1_gan_log, 'pos')
                    # dis
                    self.G_z1_dis_loss   = self.class_loss2(self.D_Z1_cls_log, self.z_lb)

                    self.G_X1_loss = (self.gen_gan_weight * self.G_z1_gan_loss )
                                    #+ self.gen_dis_weight * self.G_z1_dis_loss )

                with tf.name_scope('Domain_2'):
                    # gan
                    self.G_z2_gan_loss   = self.class_loss(self.D_Z2_gan_log, 'pos')
                    # dis
                    self.G_z2_dis_loss   = self.class_loss2(self.D_Z2_cls_log, self.z_lb)


                    self.G_X2_loss = (self.gen_gan_weight *  self.G_z2_gan_loss  )
                                    #+ self.gen_dis_weight *  self.G_z2_dis_loss )

                self.G_loss = self.G_X1_loss + self.G_X2_loss

        self.code = []
        self.z_w_code = []
        self.G1_z_w_code = []
        self.G2_z_w_code = []
        with tf.name_scope('Images_Result'):
            for i in range(self.class_num):
                self.code.append(self.opers.generate_code(i ,self.class_num,self.batch_size))

                self.z_w_code.append(self.opers.disentangle_coding(self.z, self.code[i]))

                self.G1_z_w_code.append(self.generator(self.z_w_code[i]        , 'dm1',shared_reuse=True  ,unshared_reuse=True))
                self.G2_z_w_code.append(self.generator(self.z_w_code[i]        , 'dm2',shared_reuse=True  ,unshared_reuse=True))

        with tf.name_scope('Summary'):
            self.train_sum     =  tf.summary.merge_all()
            self.train_writer  =  tf.summary.FileWriter(self.summaries_dir + '/train',self.sess.graph)

        with tf.name_scope('Variable_seperation'):
        # Extract variable from model
            total_vars = tf.trainable_variables()

            self.g_d1_vars = [var for var in total_vars if 'Generator/dm1' in var.name] + [var for var in total_vars if 'Generator/Shared' in var.name]
            self.g_d2_vars = [var for var in total_vars if 'Generator/dm2' in var.name] + [var for var in total_vars if 'Generator/Shared' in var.name]

            self.d_d1_vars = [var for var in total_vars if 'Discriminator/dm1' in var.name] + [var for var in total_vars if 'Discriminator/Shared' in var.name]
            self.d_d2_vars = [var for var in total_vars if 'Discriminator/dm2' in var.name] + [var for var in total_vars if 'Discriminator/Shared' in var.name]

            self.g_vars = [var for var in total_vars if 'Generator'     in var.name]
            self.d_vars = [var for var in total_vars if 'Discriminator' in var.name]


