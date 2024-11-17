import os
import numpy as np
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

class SpatialModel(tf.keras.models.Model):
    def __init__(self, ksize:int=32,**kwargs):
        super().__init__(**kwargs)
        self.ksize = ksize
    def build(self, input_shape):
        self.k = self.add_weight(name='kernel', shape=(self.ksize,self.ksize), dtype=tf.float32)
        pass
    def call(self, inputs):
        k = self.k[:,:,tf.newaxis, tf.newaxis]
        k = k[::-1,::-1,:,:]
        # k = tf.pad(k, [(0,1),(0,1),(0,0),(0,0)])
        out = tf.nn.conv2d(inputs, k, (1,1), padding='SAME')
        return out
    

class FreqModel(tf.keras.models.Model):
    def __init__(self, ksize:int=32,**kwargs):
        super().__init__(**kwargs)
        self.ksize = ksize
    def build(self, input_shape):
        self.k = self.add_weight(name='kernel', shape=(self.ksize,self.ksize), dtype=tf.float32)
        pass
    def get_config(self):
        config = super().get_config()
        config['ksize'] = self.ksize
        return config
    def call(self, inputs):
        img_c = tf.complex(inputs[:,:,:,0],tf.zeros_like(inputs[:,:,:,0]))
        img_f = tf.signal.fft2d(tf.signal.ifftshift(img_c))

        img_shape = tf.shape(inputs)
        k_shape = tf.shape(self.k)
        padding = tf.cast((img_shape[1:3] -k_shape)//2, tf.int32)
        k = tf.pad(self.k, [[padding[0],padding[0]],[padding[1],padding[1]]])
        k_c = tf.complex(k, tf.zeros_like(k))
        
        k_f = tf.signal.fft2d(tf.signal.ifftshift(k_c))
        out_f = k_f*img_f
        
        out = tf.signal.fftshift(tf.signal.ifft2d(out_f))
        out = tf.math.real(out)[:,:,:,tf.newaxis]
        return out

def test_fft_conv():
    N = 256
    P = 64

    img = np.zeros((1,N,N,1), np.float32)
    for y1 in range(P//2, N, P):
        for x1 in range(P//2, N, P):
            for y2 in range(-P//4,P//4):
                for x2 in range(-P//4,P//4):
                    x = x1 + x2
                    y = y1 + y2
                    img[:,y,x,:] = 1
    
    k = cv2.getGaussianKernel(32,3)
    k2 = k.reshape((-1,1))*k.reshape((1,-1))
    gt_out_img4 = cv2.filter2D(img[0,:,:,0],-1, k2)[np.newaxis, :,:,np.newaxis]

    sp_model = SpatialModel()
    sp_model.compile(optimizer='adam',loss='mse')
    sp_model.predict(img)
    freq_model = FreqModel()
    freq_model.compile(optimizer='adam',loss='mse')
    freq_model.predict(img)

    ws = sp_model.get_weights()
    # ws[0] = k2
    ws[0][:,:] = 0
    ws[0][16,16] = 1
    sp_model.set_weights(ws)
    freq_model.set_weights(ws)

    # train
    # H1 = sp_model.fit(img, gt_out_img4, epochs=10)
    H2 = freq_model.fit(img, gt_out_img4, epochs=10)
    sp_out_img4 = sp_model.predict(img)
    freq_out_img4 = freq_model.predict(img)
    fig,axs = plt.subplots(3,3,sharex=True, sharey=True)
    axs[0,0].imshow(gt_out_img4[0,:,:,0])
    axs[0,1].imshow(sp_out_img4[0,:,:,0])
    axs[0,2].imshow(freq_out_img4[0,:,:,0])
    axs[1,0].imshow(img[0,:,:,0])
    axs[1,1].imshow(sp_out_img4[0,:,:,0]-gt_out_img4[0,:,:,0], vmin=-0.1, vmax=0.1)
    axs[1,2].imshow(freq_out_img4[0,:,:,0]-gt_out_img4[0,:,:,0], vmin=-0.1, vmax=0.1)
    axs[2,0].imshow(freq_out_img4[0,:,:,0]-sp_out_img4[0,:,:,0], vmin=-0.1, vmax=0.1)
    plt.show()
    pass

if __name__ == '__main__':
    test_fft_conv()
    
