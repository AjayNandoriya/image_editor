import os
import numpy as np
import pandas as pd
import logging
LOGGER = logging.getLogger(__name__)
from matplotlib import pyplot as plt
# import primesieve
import sympy

"""
isprime(n)              # Test if n is a prime number (True) or not (False).

primerange(a, b)        # Generate a list of all prime numbers in the range [a, b).
randprime(a, b)         # Return a random prime number in the range [a, b).
primepi(n)              # Return the number of prime numbers less than or equal to n.

prime(nth)              # Return the nth prime, with the primes indexed as prime(1) = 2. The nth prime is approximately n*log(n) and can never be larger than 2**n.
prevprime(n, ith=1)     # Return the largest prime smaller than n
nextprime(n)            # Return the ith prime greater than n

sieve.primerange(a, b)  # Generate all prime numbers in the range [a, b), implemented as a dynamically growing sieve of Eratosthenes. 
"""

def create_prime_dataset(N:int=1000, out_csv:str=''):
    sympy.sieve.extend_to_no(N)
    Ps = np.array(sympy.sieve._list)
    # print(sympy.sieve._list)
    df = pd.DataFrame({
        'index':np.arange(N),
        'prime':Ps[:N]
    })
    if out_csv != '':
        df.to_csv(out_csv, index=False)
    return Ps

def test_create_prime_dataset():
    N = 10000
    out_csv = os.path.join(os.path.dirname(__file__),'primes.csv')
    create_prime_dataset(N, out_csv)



def factors(N:int, a:int=1, b:int=None, DEBUG:bool=False):
    if b is None:
        b = N

    Ps = get_Ps(100)

    N_rems = N%Ps

    if DEBUG:
        a_rems = a%Ps
        b_rems = b%Ps
        rems = np.concatenate([N_rems.reshape((1,-1)), a_rems.reshape((1,-1)), b_rems.reshape((1,-1))],axis=0)
        LOGGER.info(f'{rems[:,:20] = }')
    return a,b

def test_factors():
    gt_a,gt_b = 27791, 92377
    N = gt_a*gt_b
    a,b = factors(N, gt_a, gt_b, DEBUG=True)
    LOGGER.info(f'({gt_a} * {gt_b}) = {N}')
    LOGGER.info(f'({a} * {b}) = {a*b}')
    pass

def get_Ps(N:int):
    in_csv = os.path.join(os.path.dirname(__file__),'primes.csv')
    df= pd.read_csv(in_csv)
    return df['prime'].values[:N]


def test_mulprime():
    Ps = get_Ps(N=100)

    m = 1
    for i in range(len(Ps)):
        m = m*(Ps[i]/2)
        LOGGER.info(f'{i} : {Ps[i]} : {m}')
    pass

def test_55():
    import tensorflow as tf

    inp = tf.keras.layers.Input(shape=(None,1))
    out = tf.keras.layers.Conv1D(1,3,1,padding='same',use_bias=False)(inp)
    out = tf.keras.layers.Conv1D(1,3,1,padding='same',use_bias=False)(out)
    m3 = tf.keras.models.Model(inputs=inp, outputs=out)
    m3.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(0.05))
    inp3 = np.zeros((1,128,1),dtype=np.float32)
    inp3[0,64,0] = 1

    inp = tf.keras.layers.Input(shape=(None,1))
    l = tf.keras.layers.Conv1D(1,5,1,padding='same', use_bias=False)
    out = l(inp)
    m5 = tf.keras.models.Model(inputs=inp, outputs=out)

    ws = l.get_weights()
    ws[0] = np.array([1,2,3,4,5], dtype=np.float32).reshape((5,1,1))
    l.set_weights(ws)
    out5 = m5.predict(inp3)
    m3.fit(inp3, out5, epochs=1000)
    out3 = m3.predict(inp3)
    print(m3.trainable_variables)

    plt.plot(inp3[0,:,0],'b+')
    plt.plot(out5[0,:,0],'r+')
    plt.plot(out3[0,:,0],'g.')
    plt.grid(True)
    plt.show()
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # create_prime_dataset(10000)
    # test_create_prime_dataset()
    # test_factors()
    # test_mul5prime()
    test_55()
    pass

