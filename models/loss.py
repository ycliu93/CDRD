import tensorflow as tf
import numpy as np

class Loss:
#=========================== Loss ===============================#
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def class_loss(self,predict, label):
        if label == 'pos':
            label_mat = np.tile([0],self.batch_size)
        elif label =='neg':
            label_mat = np.tile([1],self.batch_size)
        else:
            print 'typo in class_loss'

        return tf.reduce_mean(-tf.reduce_sum(tf.log(predict + 1e-8)* tf.one_hot(indices = label_mat, depth = 2, dtype=tf.float32), 1))
    def class_loss2(self,predict, label):
        return tf.reduce_mean(-tf.reduce_sum(tf.log(predict + 1e-8)* label, 1))
    def vae_loss    (self,feature_map):

        code, noise_var   =  tf.split(feature_map,[self.input_z_size,self.input_z_size],axis = 1)
        term1             = tf.exp(noise_var)
        term2             = tf.add(tf.ones_like(noise_var) ,noise_var)
        term3             = tf.square(code)
        vae_loss          = tf.reduce_mean( tf.add(tf.subtract(term1,term2),term3))
        return vae_loss
    def percept_loss(self,result_x,y):

        loss = tf.reduce_mean(tf.square(tf.subtract(result_x, y)))
        return loss
    def tv_loss(self,tens):
        tem_x = tens[: ,1:,:,:] - tens[:,:-1,:,:]
        tem_x = tf.to_float(tf.square(tem_x))
        tem_x = tf.reduce_mean(tem_x,[1,2,3])
        tem_y = tens[: ,:, 1:,:] - tens[:,:,:-1,:]
        tem_y = tf.to_float(tf.square(tem_y))
        tem_y = tf.reduce_mean(tem_y,[1,2,3])
        tv_loss = tf.reduce_mean(tf.sqrt(tf.add(tem_x,tem_y)))
        return tv_loss
