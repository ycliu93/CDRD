import tensorflow as tf
import numpy as np
from random import shuffle, randint
import os


class Ops:
#================= Operations ====================#
    def one_hot_vec (self, label_mat):
        # label_vec : 8*1 mat     output: 8*2 tensor
        #   [1 ,0 , 1 ,1 ,1] ---->     [1 , 0 , 1 , 1, 1]
        #                              [0 , 1 , 0 , 0 ,0]
        vec1 = np.sign(label_mat )
        vec2 = -1* np.sign(label_mat -1)
        return np.concatenate((vec1,vec2),axis=1)

    def noise_coding(self,feature_map):
        z                  =  tf.random_normal([self.batch_size,self.input_z_size],
                                                mean=0.0,
                                                stddev=1.0,
                                                dtype=tf.float32,
                                                seed=None,
                                                name=None)
        code, noise_var    =  tf.split(feature_map,[self.input_z_size,self.input_z_size],axis = 1)
        ss_noise           =  tf.add(tf.multiply(tf.sqrt(tf.exp(noise_var)),z),code)
        return ss_noise
    def pass_coding (self,feature_map):
        code, _    =  tf.split(feature_map,[self.input_z_size,self.input_z_size],axis = 1)
        return code
    def sample_Z    (self, batch, Z_dim):
        mean=0.0
        stddev=1.0
        z =  np.random.normal(mean,stddev,[batch,Z_dim])
        return z
    def disentangle_coding(self, code, label):
        code_w_label = tf.concat([code, label], 1)
        return code_w_label
    def accuracy    (self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

    def gray2rgb(self,gray):
        rgb = np.concatenate ((gray,gray,gray), axis=1)
        return rgb

    def shuffle_train_idx(self,length):
        x = [i for i in range(length)]
        shuffle(x)
        return x

    def regular_train_idx(self,length):
        x = [i for i in range(length)]
        return x

    def get_batch(self,dataset,batch_size,idx,k):
        batch = np.array([dataset[idx[i+batch_size*k]]  for i in range(batch_size)])
        return batch

    def compact_batch_img(self,input_npary,input_npary2,input_npary3,input_npary4,input_npary5):
        if input_npary.shape[3]==1:
            tmp = np.concatenate( (input_npary, input_npary, input_npary), axis=3)
        else:
            tmp = input_npary
        tmp  = np.multiply(tmp, 128.)
        tmp  = np.add(tmp, 128.)

        if input_npary2.shape[3]==1:
            tmp2 = np.concatenate( (input_npary2, input_npary2, input_npary2), axis=3)
        else:
            tmp2 = input_npary2
        tmp2  = np.multiply(tmp2, 128.)
        tmp2  = np.add(tmp2, 128.)

        if input_npary3.shape[3]==1:
            tmp3 = np.concatenate( (input_npary3, input_npary3, input_npary3), axis=3)
        else:
            tmp3 = input_npary3
        tmp3  = np.multiply(tmp3, 128.)
        tmp3  = np.add(tmp3, 128.)

        if input_npary4.shape[3]==1:
            tmp4 = np.concatenate( (input_npary4, input_npary4, input_npary4), axis=3)
        else:
            tmp4 = input_npary4
        tmp4  = np.multiply(tmp4, 128.)
        tmp4  = np.add(tmp4, 128.)

        if input_npary5.shape[3]==1:
            tmp5 = np.concatenate( (input_npary5, input_npary5, input_npary5), axis=3)
        else:
            tmp5 = input_npary5
        tmp5  = np.multiply(tmp5, 128.)
        tmp5  = np.add(tmp5, 128.)
        row1 = np.concatenate( ( tmp[0,:,:,:], tmp[1,:,:,:], tmp[2,:,:,:], tmp[3,:,:,:], tmp[4,:,:,:], tmp[5,:,:,:], tmp[6,:,:,:], tmp[7,:,:,:]),axis=1)
        row2 = np.concatenate( (tmp2[0,:,:,:],tmp2[1,:,:,:],tmp2[2,:,:,:],tmp2[3,:,:,:],tmp2[4,:,:,:],tmp2[5,:,:,:],tmp2[6,:,:,:],tmp2[7,:,:,:]),axis=1)
        row3 = np.concatenate( (tmp3[0,:,:,:],tmp3[1,:,:,:],tmp3[2,:,:,:],tmp3[3,:,:,:],tmp3[4,:,:,:],tmp3[5,:,:,:],tmp3[6,:,:,:],tmp3[7,:,:,:]),axis=1)
        row4 = np.concatenate( (tmp4[0,:,:,:],tmp4[1,:,:,:],tmp4[2,:,:,:],tmp4[3,:,:,:],tmp4[4,:,:,:],tmp4[5,:,:,:],tmp4[6,:,:,:],tmp4[7,:,:,:]),axis=1)
        row5 = np.concatenate( (tmp5[0,:,:,:],tmp5[1,:,:,:],tmp5[2,:,:,:],tmp5[3,:,:,:],tmp5[4,:,:,:],tmp5[5,:,:,:],tmp5[6,:,:,:],tmp5[7,:,:,:]),axis=1)
        compact_img = np.concatenate( (row1,row2,row3,row4,row5),axis=0)
        return compact_img
    def compact_batch_img2(self,input_npary,input_npary2,input_npary3,input_npary4):
        if input_npary.shape[3]==1:
            tmp = np.concatenate( (input_npary, input_npary, input_npary), axis=3)
        else:
            tmp = input_npary
        tmp  = np.multiply(tmp, 256.)
        tmp  = np.add(tmp, 128.)
        if input_npary2.shape[3]==1:
            tmp2 = np.concatenate( (input_npary2, input_npary2, input_npary2), axis=3)
        else:
            tmp2 = input_npary2
        tmp2  = np.multiply(tmp2, 256.)
        tmp2  = np.add(tmp2, 128.)

        if input_npary3.shape[3]==1:
            tmp3 = np.concatenate( (input_npary3, input_npary3, input_npary3), axis=3)
        else:
            tmp3 = input_npary3
        tmp3  = np.multiply(tmp3, 256.)
        tmp3  = np.add(tmp3, 128.)

        if input_npary4.shape[3]==1:
            tmp4 = np.concatenate( (input_npary4, input_npary4, input_npary4), axis=3)
        else:
            tmp4 = input_npary4
        tmp4  = np.multiply(tmp4, 256.)
        tmp4  = np.add(tmp4, 128.)
        row1 = np.concatenate( ( tmp[0,:,:,:], tmp[1,:,:,:], tmp[2,:,:,:], tmp[3,:,:,:], tmp[4,:,:,:], tmp[5,:,:,:], tmp[6,:,:,:], tmp[7,:,:,:]),axis=1)
        row2 = np.concatenate( (tmp2[0,:,:,:],tmp2[1,:,:,:],tmp2[2,:,:,:],tmp2[3,:,:,:],tmp2[4,:,:,:],tmp2[5,:,:,:],tmp2[6,:,:,:],tmp2[7,:,:,:]),axis=1)
        row3 = np.concatenate( (tmp3[0,:,:,:],tmp3[1,:,:,:],tmp3[2,:,:,:],tmp3[3,:,:,:],tmp3[4,:,:,:],tmp3[5,:,:,:],tmp3[6,:,:,:],tmp3[7,:,:,:]),axis=1)
        row4 = np.concatenate( (tmp4[0,:,:,:],tmp4[1,:,:,:],tmp4[2,:,:,:],tmp4[3,:,:,:],tmp4[4,:,:,:],tmp4[5,:,:,:],tmp4[6,:,:,:],tmp4[7,:,:,:]),axis=1)
        compact_img = np.concatenate( (row1,row2,row3,row4),axis=0)
        return compact_img

    def compact_row_img(self,input_npary):
        if input_npary.shape[3]==1:
            tmp = np.concatenate( (input_npary, input_npary, input_npary), axis=3)
        else:
            tmp = input_npary
        tmp  = np.multiply(tmp, 256.)
        tmp  = np.add(tmp, 128.)
        row = np.concatenate( ( tmp[0,:,:,:], tmp[1,:,:,:], tmp[2,:,:,:], tmp[3,:,:,:], tmp[4,:,:,:], tmp[5,:,:,:], tmp[6,:,:,:], tmp[7,:,:,:]),axis=1)
        return row

    def compact_batch_img3(self,input_dm1,input_dm2, input_npary,input_npary2,input_npary3,input_npary4,input_npary5,input_npary6,input_npary7,input_npary8,input_npary9,input_npary10,input_npary11,input_npary12,input_npary13,input_npary14,input_npary15,input_npary16,input_npary17,input_npary18,input_npary19,input_npary20):
        #input1,2,3 : [-0.5,0.5] rgb or gray
        row_dm1 = self.compact_row_img(input_dm1)
        row_dm2 = self.compact_row_img(input_dm2)
        row1 = self.compact_row_img(input_npary)
        row2 = self.compact_row_img(input_npary2)
        row3 = self.compact_row_img(input_npary3)
        row4 = self.compact_row_img(input_npary4)
        row5 = self.compact_row_img(input_npary5)
        row6 = self.compact_row_img(input_npary6)
        row7 = self.compact_row_img(input_npary7)
        row8 = self.compact_row_img(input_npary8)
        row9 = self.compact_row_img(input_npary9)
        row10= self.compact_row_img(input_npary10)
        compact_img1 = np.concatenate( (row1,row2,row3,row4,row5,row6,row7,row8,row9,row10),axis=0)
        row11 = self.compact_row_img(input_npary11)
        row12 = self.compact_row_img(input_npary12)
        row13 = self.compact_row_img(input_npary13)
        row14 = self.compact_row_img(input_npary14)
        row15 = self.compact_row_img(input_npary15)
        row16 = self.compact_row_img(input_npary16)
        row17 = self.compact_row_img(input_npary17)
        row18 = self.compact_row_img(input_npary18)
        row19 = self.compact_row_img(input_npary19)
        row20 = self.compact_row_img(input_npary20)
        compact_img2 = np.concatenate( (row11,row12,row13,row14,row15,row16,row17,row18,row19,row20),axis=0)
        compact_img = np.concatenate( (compact_img1, compact_img2),axis=1)
        return compact_img

    def pre_process(self,input_npary, if_random_flop):
        tmp = np.subtract(input_npary,0.5)
        tmp = tmp*2
        ## random flip
        if if_random_flop:
            rand = randint(0,99)
            if rand > 50:
                tmp = np.flip(tmp, 3)
        return tmp

    def idx2one_hot(self,size, cls_num, input_):
        one_hot = np.zeros((size, cls_num))
        one_hot[np.arange(size), input_] = 1
        return one_hot

    def one_hot2idx(self, input_):
        idx = np.argmax(input_, axis=1)
        return idx

    def random_one_hot(self,cls_num, batch_size):
        one_hot = np.eye(cls_num)[np.random.choice(cls_num, batch_size)]
        index   = np.argmax(one_hot,1)
        return one_hot, index

    def generate_code(self, idx ,cls_num, batch_size):
        idx_vector = np.ones(batch_size) * idx
        idx_vector = idx_vector.astype(int)
        zero_mat   = np.zeros((batch_size, cls_num))
        zero_mat[np.arange(batch_size), idx_vector ] = 1
        return zero_mat

    def record_parameter(self):
        with open(self.sample_path+"Parameter .txt", "a") as text_file:
            text_file.write("====== Loss Weight ======= \n" )
            text_file.write("gen_gan_weight: %f \n"  % (self.gen_gan_weight))
            text_file.write("gen_dis_weight: %f \n"  % (self.gen_dis_weight))
            text_file.write("dis_gan_weight: %f \n"  % (self.dis_gan_weight))
            text_file.write("dis_dis_weight: %f \n\n"  % (self.dis_dis_weight))
            text_file.write("use_mse_loss: %d \n"  % (self.use_mse_loss))
            text_file.write("mse_precept_weight: %f \n\n"  % (self.mse_precept_weight))

            text_file.write("use_tv_loss: %d \n"  % (self.use_tv_loss))
            text_file.write("tv_weight: %f \n\n"  % (self.tv_weight))
            text_file.write("====== GAN Update Times ======= \n")

            text_file.write("G_GAN_update_times: %d \n"  % (self.G_GAN_update_times))
            text_file.write("D_GAN_update_times: %d \n\n"  % (self.D_GAN_update_times))
            text_file.write("====== Learning Rate ======= \n" )
