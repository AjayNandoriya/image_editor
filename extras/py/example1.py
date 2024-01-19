import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


def test_downsample():
    N = 8
    l = tf.keras.layers.AveragePooling2D()
    inp = np.arange(N**2, dtype=np.float32).reshape((1,N,N,1))
    # inp = np.zeros((1,N,N,1), dtype=np.float32)
    # inp[0,0,0,0] = 1
    # inp[0,N-1,0,0] = 1
    # inp[0,0,N-1,0] = 1
    # inp[0,N-1,N-1,0] = 1
    out = l(inp)

    plt.subplot(121),plt.imshow(inp[0,:,:,0])
    plt.subplot(122),plt.imshow(out[0,:,:,0])
    plt.show()
    pass

def test_upsample():
    N = 8
    l = tf.keras.layers.UpSampling2D(interpolation='bilinear')
    inp = np.arange(N**2, dtype=np.float32).reshape((1,N,N,1))
    out = l(inp)

    plt.subplot(121),plt.imshow(inp[0,:,:,0])
    plt.subplot(122),plt.imshow(out[0,:,:,0])
    plt.show()
    pass

def test_convt():
    N = 16
    l = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=(5,5), strides=(2,2),padding='same', use_bias=False)
    l.build((1,8,8,1))
    ws = l.get_weights()
    ws[0] = np.array([[0.25, 0.5, 0.25,0],[0.5, 1.0, 0.5,0],[0.25, 0.5, 0.25,0],[0, 0, 0,0]]).reshape((4,4,1,1))
    ws[0] = np.array([[1, 3, 3, 1, 0],
                      [3, 9, 9, 3, 0],
                      [3, 9, 9, 3, 0],
                      [1, 3, 3, 1, 0],
                      [0, 0, 0, 0,0]], dtype=np.float32).reshape((5,5,1,1))/16
    l.set_weights(ws)

    l1 = tf.keras.layers.AveragePooling2D()
    inp = np.arange(N**2, dtype=np.float32).reshape((1,N,N,1))

    inp = np.random.rand(1,N,N,1)
    out2 = l1(inp)
    out = l(out2)
    inp = out
    out2 = l1(inp)
    out = l(out2)

    plt.subplot(131),plt.imshow(inp[0,:,:,0],vmin=0, vmax=1)
    plt.subplot(132),plt.imshow(out[0,:,:,0],vmin=0, vmax=1)
    plt.subplot(133),plt.imshow(out[0,:,:,0]- inp[0,:,:,0],vmin=-1, vmax=1)
    plt.show()
    pass


def bicubic(t:float=0):
    T = np.array([[1, t, t**2, t**3]], dtype=np.float32)/2
    A = np.array([[0,2,0,0],
                  [-1,0,1,0],
                  [2,-5,4,-1],
                  [-1,3,-3,1]], dtype=np.float32)
    B = np.matmul(T,A)
    return B

def test_bicubic():
    b = bicubic(t=0.5)
    print(b)
    b = bicubic(t=0.25)
    print(b)
    b = bicubic(t=0.75)
    print(b)
    pass

def test_downsample2():
    d2 = tf.keras.layers.Conv2D(1, (4,4), strides=(2,2), padding='same', use_bias=False)
    d2.build((1,8,8,1))
    ws = d2.get_weights()

    A = np.array([-0.0625, 0.5625, 0.5625, -0.0625],dtype=np.float32).reshape((1,4))
    At = A.T
    AA = np.matmul(At,A)
    ws[0] = AA.reshape((4,4,1,1))
    d2.set_weights(ws)

    u2 = tf.keras.layers.Conv2DTranspose(1,(8,8), strides=(2,2),padding='same', use_bias=False)
    u2.build((1,8,8,1))
    ws = u2.get_weights()
    
    A = np.array([-0.0234375, -0.0703125, 0.2265625, 0.8671875, 0.8671875, 0.2265625, -0.0703125,-0.0234375],dtype=np.float32).reshape((1,8))
    At = A.T
    AA = np.matmul(At,A)
    ws[0] = AA.reshape((8,8,1,1))
    u2.set_weights(ws)


    u1 = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=(5,5), strides=(2,2),padding='same', use_bias=False)
    u1.build((1,8,8,1))
    ws = u1.get_weights()
    ws[0] = np.array([[0.25, 0.5, 0.25,0],[0.5, 1.0, 0.5,0],[0.25, 0.5, 0.25,0],[0, 0, 0,0]]).reshape((4,4,1,1))
    ws[0] = np.array([[1, 3, 3, 1, 0],
                      [3, 9, 9, 3, 0],
                      [3, 9, 9, 3, 0],
                      [1, 3, 3, 1, 0],
                      [0, 0, 0, 0,0]], dtype=np.float32).reshape((5,5,1,1))/16
    u1.set_weights(ws)

    d1 = tf.keras.layers.AveragePooling2D()


    N = 32
    inp = np.random.rand(1,N,N,1)
    # out2 = d2(inp)
    # out = u2(out2)
    # inp = out
    # out2 = d2(inp)
    # out = u2(out2)
    # inp = out
    out2 = d2(inp)
    out2 = u2(out2)
    diff2 = out2-inp

    SNR2 = np.std(diff2)/np.std(inp)
    plt.subplot(331),plt.imshow(inp[0,:,:,0],vmin=0, vmax=1)
    plt.subplot(332),plt.imshow(out2[0,:,:,0],vmin=0, vmax=1)
    plt.subplot(333),plt.imshow(out2[0,:,:,0]- inp[0,:,:,0],vmin=-1, vmax=1)
    plt.title(f'{SNR2 = }')

    out1 = d1(inp)
    out1 = u1(out1)
    diff1 = out1-inp
    SNR1 = np.std(diff1)/np.std(inp)
    plt.subplot(334),plt.imshow(inp[0,:,:,0],vmin=0, vmax=1)
    plt.subplot(335),plt.imshow(out1[0,:,:,0],vmin=0, vmax=1)
    plt.subplot(336),plt.imshow(out1[0,:,:,0]- inp[0,:,:,0],vmin=-1, vmax=1)
    plt.title(f'{SNR1 = }')
    plt.show()

if __name__ == '__main__':
    # test_downsample()
    # test_upsample()
    # test_convt()

    # test_bicubic()
    test_downsample2()
    pass