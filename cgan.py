from __future__ import print_function, division

from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM, Activation

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

import numpy as np


class Modelos_DGAN(object):
    def __init__(self, look_back, vars, var_y):

        self.look_back = look_back      
        self.vars = vars  
        self.var_y = var_y  
        self.esqueleto_D = None  
        self.esqueleto_G = None 
        self.modelo_A = None  
        self.modelo_D = None 

    def esqueleto_discriminator(self):
        if self.esqueleto_D: 
            return self.esqueleto_D
        
        
        self.esqueleto_D = Sequential()

        self.esqueleto_D.add(Dense(128, input_dim=self.var_y, activation='relu', kernel_initializer='he_normal'))
        self.esqueleto_D.add(Dropout(0.5))  
        self.esqueleto_D.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        self.esqueleto_D.add(Dense(1, activation='sigmoid'))
        
        return self.esqueleto_D

    def esqueleto_generator(self):
        if self.esqueleto_G:
            return self.esqueleto_G
        self.esqueleto_G = Sequential()
        self.esqueleto_G.add(LSTM(64, input_shape=(self.look_back, self.vars), dropout=0.2, recurrent_dropout=0.2))
        self.esqueleto_G.add(Dense(32, activation='relu', kernel_regularizer='l2'))
        self.esqueleto_G.add(Dropout(0.5))
        self.esqueleto_G.add(Dense(self.var_y, activation='linear'))  

        return self.esqueleto_G

    def modelo_discriminator(self): 
        if self.modelo_D:
            return self.modelo_D
        optimizer = Adam(lr=0.00002)
        self.modelo_D = Sequential()
        self.modelo_D.add(self.esqueleto_discriminator())
        self.modelo_D.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.modelo_D 

    def modelo_adversarial(self): 
        if self.modelo_A:
            return self.modelo_A
        optimizer = Adam(lr=0.00001)
        self.modelo_A = Sequential()
        self.modelo_A.add(self.esqueleto_generator())
        self.modelo_A.add(self.esqueleto_discriminator())
        self.modelo_A.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.modelo_A
    



class DCGAN(object):
    def __init__(self, x_treino=None, look_back = 3, vars = None, var_y= None):
        self.look_back = look_back

        self.x_treino = x_treino[:,3]
        self.vars = vars
        self.var_y = var_y

        self.modelos_DCGAN = Modelos_DGAN(look_back, vars, var_y)
        self.discriminator =  self.modelos_DCGAN.modelo_discriminator()
        self.adversarial = self.modelos_DCGAN.modelo_adversarial()
        self.generator = self.modelos_DCGAN.esqueleto_generator()
        
    
    def discriminator_trainable(self, val): 
        self.discriminator.trainable = val
        for l in self.discriminator.layers:
            l.trainable = val
    
    
    def create_dataset(self, dataset, look_back=1):
        dataX = []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), :]
            dataX.append(a)
        return np.array(dataX)

    
    def train(self, train_steps=5000, batch_size=250):

        for i in range(train_steps):
            
            dados_reais = self.x_treino[np.random.randint(0, self.x_treino.shape[0], size=batch_size), ] 
            dados_reais = dados_reais.reshape((dados_reais.shape[0],1))
            ruidos = np.random.uniform(0, 1.0, size=[batch_size+self.look_back+1, self.vars]) 
            ruidos = self.create_dataset(ruidos,self.look_back)
            dados_fake = self.generator.predict(ruidos) 
            x = np.concatenate((dados_reais, dados_fake)) 
            
            
            y = np.ones([2*batch_size, 1])  
            y[batch_size:, :] = 0
            
            
            self.discriminator_trainable(True)
            loss_discriminator = self.discriminator.train_on_batch(x, y) 

            
            y = np.ones([batch_size, 1])
            ruidos = np.random.uniform(0, 1.0, size=[batch_size+self.look_back+1, self.vars]) 
            ruidos = self.create_dataset(ruidos,self.look_back)
            self.discriminator_trainable(False)
            loss_adversarial = self.adversarial.train_on_batch(ruidos, y)
            
               
            print('%d: [Discriminator loss: %f, acc: %f]   [Adversarial loss: %f, acc: %f]'  % (i, loss_discriminator[0], 
                                                                                                loss_discriminator[1], 
                                                                                                loss_adversarial[0], 
                                                                                                loss_adversarial[1]))
            
    
