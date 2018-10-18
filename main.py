import tensorflow as tf
import numpy as np
import scipy.misc
import random
from random import shuffle, randint
import sys
import os
import random
import gzip
import time
import pickle
import argparse
import models
from utils.ops import Ops
from models.model import Model
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class CDRD:
    def __init__(self,sess,args):
        self.start_time =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.sess = sess

        if not os.path.exists(args.sample_path):
            os.makedirs(args.sample_path)
            os.makedirs(args.sample_path+'Z/')
            os.makedirs(args.sample_path+'dm1/')

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        self.G_model_filepath   = args.model_path  + 'G.ckpt'
        self.D_model_filepath   = args.model_path  + 'D.ckpt'


        if args.status=='test' :
            if not os.path.exists(args.sample_path+args.restore_num+'/'):
                os.makedirs(args.sample_path+args.restore_num+'/')
                os.makedirs(args.sample_path+'Z/')
                os.makedirs(args.sample_path+'dm1/')

            self.G_model_filepath   = self.G_model_filepath + args.restore_num
            self.D_model_filepath   = self.D_model_filepath + args.restore_num
        self.ops   = Ops()
        self.model = Model(sess,args.summaries_path, args.batch_size, args.input_z_size,args.emb_size, args.class_num, args.code_dim, args.img_size)
        self.train(args)

 #============== Train ==================#
    def train(self,args):

        # Load Data
        with gzip.open(args.data_dm1_path, 'rb') as f:
            dm1_train_set, dm1_valid_set, dm1_test_set = pickle.load(f)

        self.dm1_train, self.dm1_lb_train = dm1_train_set
        self.dm1_test,  self.dm1_lb_test  = dm1_test_set
        self.dm1_val,   self.dm1_lb_val   = dm1_valid_set
        # MNIST choose 2000 ; USPS choose 1800

        self.dm1_train    = np.concatenate( (self.dm1_train, self.dm1_val)      , axis=0)
        self.dm1_lb_train = np.concatenate( (self.dm1_lb_train, self.dm1_lb_val), axis=0)

        self.dm1_train_choose_idx = np.random.choice(len(self.dm1_train), 2000, replace=False)
        self.dm1_test_choose_idx = np.random.choice(len(self.dm1_test), 2000, replace=False)
        self.dm1_train = self.dm1_train[self.dm1_train_choose_idx]
        self.dm1_test  = self.dm1_test[self.dm1_test_choose_idx]
        self.dm1_lb_train = self.dm1_lb_train[self.dm1_train_choose_idx]
        self.dm1_lb_test  = self.dm1_lb_test[self.dm1_test_choose_idx]

        self.dm1_train = self.dm1_train.reshape((-1, 1,28,28))
        self.dm1_test  = self.dm1_test.reshape((-1, 1,28,28))
        self.dm1_lb_train = self.ops.idx2one_hot(self.dm1_lb_train.shape[0], args.class_num, self.dm1_lb_train)
        self.dm1_lb_test  = self.ops.idx2one_hot(self.dm1_lb_test.shape[0],args.class_num, self.dm1_lb_test)

        # For GAN dm1
        self.dm1_2_train_choose_idx = np.random.choice(len(self.dm1_train), 2000, replace=False)
        self.dm1_2_train            = self.dm1_train[self.dm1_2_train_choose_idx]
        self.dm1_2_train            = self.dm1_2_train.reshape((-1, 1,28,28))

        with gzip.open(args.data_dm2_path) as f:
            dm2_train_set,dm2_test_set = pickle.load(f)

            self.dm2_train, self.dm2_lb_train = dm2_train_set
            self.dm2_test , self.dm2_lb_test  = dm2_test_set

        self.dm2_train_choose_idx = np.random.choice(len(self.dm2_train), 1800, replace=False)
        self.dm2_test_choose_idx = np.random.choice(len(self.dm2_test), 1800, replace=False)
        self.dm2_train = self.dm2_train[self.dm2_train_choose_idx]
        self.dm2_test  = self.dm2_test[self.dm2_test_choose_idx]
        self.dm2_lb_train = self.dm2_lb_train[self.dm2_train_choose_idx]
        self.dm2_lb_test  = self.dm2_lb_test[self.dm2_test_choose_idx]

        self.dm2_lb_train = self.ops.idx2one_hot(self.dm2_lb_train.shape[0],args.class_num, self.dm2_lb_train)
        self.dm2_lb_test  = self.ops.idx2one_hot(self.dm2_lb_test.shape[0],args.class_num, self.dm2_lb_test)

        # For GAN dm2
        self.dm2_2_train_choose_idx = np.random.choice(len(self.dm2_train), 1800, replace=False)
        self.dm2_2_train            = self.dm2_train[self.dm2_2_train_choose_idx]
        self.dm2_2_train            = self.dm2_2_train.reshape((-1, 1,28,28))

        # Iteration number and data number
        self.dm1_im_train_num              =  len(self.dm1_train)
        self.dm1_im_train_iter_num         =  self.dm1_im_train_num / args.batch_size

        self.dm1_2_im_train_num            =  len(self.dm1_2_train)
        self.dm1_2_im_train_iter_num       =  self.dm1_2_im_train_num / args.batch_size

        self.dm2_im_train_num              =  len(self.dm2_train)
        self.dm2_im_train_iter_num         =  self.dm2_im_train_num / args.batch_size

        self.dm2_2_im_train_num            =  len(self.dm2_2_train)
        self.dm2_2_im_train_iter_num       =  self.dm2_2_im_train_num / args.batch_size

        self.dm1_im_test_num               =  len(self.dm1_test)
        self.dm1_im_test_iter_num          =  self.dm1_im_test_num / args.batch_size

        self.dm2_im_test_num               =  len(self.dm2_test)
        self.dm2_im_test_iter_num          =  self.dm2_im_test_num / args.batch_size

        ## Opt.
        with tf.name_scope('Optimizer'):
            with tf.name_scope('Discriminator'):
                with tf.name_scope('Domain_1'):
                    self.dm1_d_gan_train_op = tf.train.AdamOptimizer(learning_rate=args.learning_rate,
                                      beta1=0.5,
                                      beta2=0.999,
                                      epsilon=1e-08,
                                      name='Adam1').minimize(self.model.D_loss+self.model.D_loss2 , var_list=self.model.d_d1_vars + self.model.d_d2_vars )

                    self.dm1_d_gan_train_op2 = tf.train.AdamOptimizer(learning_rate=args.learning_rate,
                                      beta1=0.5,
                                      beta2=0.999,
                                      epsilon=1e-08,
                                      name='Adam2').minimize(self.model.D_loss+self.model.D_loss2 , var_list= self.model.d_d1_vars + self.model.g_d1_vars + self.model.d_d2_vars + self.model.g_d2_vars )

            with tf.name_scope('Generator'):
                with tf.name_scope('Domain_1'):
                    self.dm1_g_train_op = tf.train.AdamOptimizer(learning_rate=args.learning_rate ,
                                      beta1=0.5,
                                      beta2=0.999,
                                      epsilon=1e-08,
                                      name='Adam3').minimize(self.model.G_loss+self.model.D_loss2 , var_list=self.model.g_d1_vars + self.model.g_d2_vars )

        with tf.name_scope('Initial'):
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            # restore the model(parameter )
        self.G_saver  = tf.train.Saver(var_list=self.model.g_vars, max_to_keep=2)
        self.D_saver  = tf.train.Saver(var_list=self.model.d_vars, max_to_keep=2)
        print('Saver Build!')

        if args.restore_mode == True:
            self.G_saver.restore(self.sess, self.G_model_filepath)
            print("G Model restored in file: %s" % self.G_model_filepath)
            print('Restore G!')
            self.D_saver.restore(self.sess, self.D_model_filepath)
            print("D Model restored in file: %s" % self.D_model_filepath)
            print('Restore D!')

        # =========== Start Train and Val. =========== #
        if args.status == 'train':
            for i in range(0,args.epoch):
                    sample_z = self.ops.sample_Z (args.batch_size,args.input_z_size)
                    sample_z_lb,_ = self.ops.random_one_hot(args.class_num, args.batch_size)
                    self.dm1_1_img_train_idx     =  self.ops.shuffle_train_idx(self.dm1_im_train_num)

                    for dm1_k in range(0,self.dm1_im_train_iter_num):

                        stage = dm1_k+i*self.dm1_im_train_iter_num

                        self.dm1_2_im_train_iter_num       =  self.dm1_2_im_train_num / args.batch_size
                        self.dm2_im_train_iter_num         =  self.dm2_im_train_num   / args.batch_size
                        self.dm2_2_im_train_iter_num       =  self.dm2_2_im_train_num / args.batch_size

                        dm2_k   = dm1_k % self.dm2_im_train_iter_num
                        dm1_2_k = dm1_k % self.dm1_2_im_train_iter_num
                        dm2_2_k = dm1_k % self.dm2_2_im_train_iter_num

                        if dm2_k == 0 :
                            self.dm2_1_img_train_idx  =  self.ops.shuffle_train_idx(self.dm2_im_train_num)
                        if dm1_2_k == 0 :
                            self.dm1_2_img_train_idx  =  self.ops.shuffle_train_idx(self.dm1_2_im_train_num)
                        if dm2_2_k == 0 :
                            self.dm2_2_img_train_idx  =  self.ops.shuffle_train_idx(self.dm2_2_im_train_num)

                        # For disentanglement - dm1
                        batch_dm1_1_img    = self.ops.pre_process(self.ops.get_batch(self.dm1_train        , args.batch_size, self.dm1_1_img_train_idx , dm1_k   ), if_random_flop= False).transpose(0,2,3,1)
                        batch_dm1_lb       =                     (self.ops.get_batch(self.dm1_lb_train     , args.batch_size, self.dm1_1_img_train_idx , dm1_k   ))

                        # For disentanglement - dm2
                        batch_dm2_1_img    = self.ops.pre_process(self.ops.get_batch(self.dm2_train        ,  args.batch_size, self.dm2_1_img_train_idx , dm2_k   ), if_random_flop= False).transpose(0,2,3,1)
                        batch_dm2_lb       =                     (self.ops.get_batch(self.dm2_lb_train     ,  args.batch_size, self.dm2_1_img_train_idx , dm2_k   ))

                        # Update Encoder & Generator
                        for _ in range(args.update_time):
                            _,G_loss = self.sess.run([self.dm1_g_train_op,self.model.G_loss],
                                            feed_dict = {self.model.X1        : batch_dm1_1_img,
                                                         self.model.X2        : batch_dm2_1_img,
                                                         self.model.X1_lb     : batch_dm1_lb,
                                                         self.model.X2_lb     : batch_dm2_lb,
                                                         self.model.z         : sample_z,
                                                         self.model.z_lb      : sample_z_lb })
                        _,D_gan_loss,D_dis_loss= self.sess.run([self.dm1_d_gan_train_op,self.model.D_loss,self.model.D_loss2],
                                            feed_dict = {self.model.X1        : batch_dm1_1_img,
                                                         self.model.X2        : batch_dm2_1_img,
                                                         self.model.X1_lb     : batch_dm1_lb,
                                                         self.model.X2_lb     : batch_dm2_lb,
                                                         self.model.z         : sample_z,
                                                         self.model.z_lb      : sample_z_lb })
                        # Summary
                        train_sum = self.sess.run(self.model.train_sum,
                                            feed_dict = {self.model.X1        : batch_dm1_1_img,
                                                         self.model.X2        : batch_dm2_1_img,
                                                         self.model.X1_lb     : batch_dm1_lb,
                                                         self.model.X2_lb     : batch_dm2_lb,
                                                         self.model.z         : sample_z,
                                                         self.model.z_lb      : sample_z_lb })

                        if stage % 5 == 0:
                            print 'Iter: %d  D dis: %f  gan: %f  G : %f ' %( stage, D_dis_loss, D_gan_loss, G_loss)
                        if stage % args.val_fre == 0:
                            self.dm1_img_test_idx     =  self.ops.regular_train_idx(self.dm1_im_test_num)
                            self.dm2_img_test_idx     =  self.ops.regular_train_idx(self.dm2_im_test_num)
                            D1_total_acc = 0
                            D2_total_acc = 0
                            D1_emb_all             = np.zeros((0, args.emb_size))
                            D2_emb_all             = np.zeros((0, args.emb_size))

                        ## Save weight
                        if (stage)% args.val_fre == 0:
                            if args.save_mode == True:
                                G_save_path = self.G_saver.save(self.sess, self.G_model_filepath, global_step=(dm1_k+i*self.dm1_im_train_iter_num))
                                print("G Model saved in file: %s" % G_save_path)
                                D_save_path = self.D_saver.save(self.sess, self.D_model_filepath, global_step=(dm1_k+i*self.dm1_im_train_iter_num))
                                print("D Model saved in file: %s" % D_save_path)

                        if stage%args.val_fre==0:
                            self.test(stage,args)
        else:
            self.test()
    def test(self,stage,args):
        self.dm1_1_img_test_idx     =  self.ops.shuffle_train_idx(self.dm1_im_test_num)
        self.dm2_1_img_test_idx     =  self.ops.shuffle_train_idx(self.dm2_im_test_num)

        sample_z           = self.ops.sample_Z (args.batch_size,args.input_z_size)
        batch_dm1_1_img    = self.ops.pre_process(self.ops.get_batch(self.dm1_test        ,  args.batch_size, self.dm1_1_img_test_idx , 0   ), if_random_flop= False).transpose(0,2,3,1)
        batch_dm2_1_img    = self.ops.pre_process(self.ops.get_batch(self.dm2_test        ,  args.batch_size, self.dm2_1_img_test_idx , 0   ), if_random_flop= False).transpose(0,2,3,1)

        G1_z_0, G1_z_1, G1_z_2, G1_z_3,G1_z_4,G1_z_5,G1_z_6,G1_z_7,G1_z_8,G1_z_9, G2_z_0, G2_z_1, G2_z_2, G2_z_3,G2_z_4,G2_z_5,G2_z_6,G2_z_7,G2_z_8,G2_z_9 = self.sess.run(
                self.model.G1_z_w_code[0:10]+self.model.G2_z_w_code[0:10]
                ,feed_dict = {self.model.z       : sample_z })

        X1, X2 = self.sess.run(
                [self.model.X1, self.model.X2]
                ,feed_dict = {self.model.X1       : batch_dm1_1_img,
                              self.model.X2       : batch_dm2_1_img })

        G1_z_0, G1_z_1 = self.sess.run(self.model.G1_z_w_code[0:2],feed_dict = {self.model.z       : sample_z })

        compact_Z        = self.ops.compact_batch_img3(X1,X2,G1_z_0, G1_z_1, G1_z_2, G1_z_3,G1_z_4,G1_z_5,G1_z_6,G1_z_7,G1_z_8,G1_z_9,G2_z_0, G2_z_1, G2_z_2, G2_z_3,G2_z_4,G2_z_5,G2_z_6,G2_z_7,G2_z_8,G2_z_9)
        scipy.misc.imsave(args.sample_path+'Z/'+str(stage)+''+'.jpg', compact_Z)

        compact_Z        = self.ops.compact_batch_img(X1,X2,G1_z_0,G1_z_1,X1)
        scipy.misc.imsave(args.sample_path+'dm1/'+str(stage)+''+'.jpg', compact_Z)

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='CDRD')
    parser.add_argument('--data_dm1_path', type=str, default='data/digit/mnist.pkl.gz')
    parser.add_argument('--data_dm2_path', type=str, default='data/digit/usps_28x28.pkl')
    parser.add_argument('--summaries_path', type=str, default='log/')
    parser.add_argument('--sample_path', type=str, default= 'sample/')
    parser.add_argument('--model_path', type=str, default='weight/')
    parser.add_argument('--status', type=str, default='train')
    parser.add_argument('--restore_num', type=str, default='-0')

    parser.add_argument('--restore_mode', type=bool, default=False)
    parser.add_argument('--save_mode', type=bool, default=True)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--input_z_size', type=int, default=100)
    parser.add_argument('--code_dim', type=int, default=10)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--emb_size', type=int, default=500)
    parser.add_argument('--update_time', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=8e-5)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--tsne_fre', type=int, default=3000)
    parser.add_argument('--val_fre',type=int, default=1000)
    parser.add_argument('--use_batch_norm', type=bool, default=False)
    args = parser.parse_args(argv)
    return args
def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

   # Launch the graph
    with tf.Session(config = config) as sess:
        cdrd = CDRD(sess)
        cdrd.train()

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        cdrd = CDRD(sess,get_args(sys.argv[1:]))
        #cdrd.train()
