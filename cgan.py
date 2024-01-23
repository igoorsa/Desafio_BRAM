from __future__ import print_function, division

#Importando as bibliotecas

from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

import numpy as np

#Criando um modelo GAN 
class Modelos_DGAN(object):

    def __init__(self, look_back, var, var_y):
        

        self.look_back = look_back      
        self.var = var  
        self.var_y = var_y  

        self.esqueleto_D = None  
        self.esqueleto_G = None 
        self.modelo_A = None  
        self.modelo_D = None 

    #Esqueleto discriminator
        #Ele vai verificar se o dado gerado é um dado real ou previsto
    def esqueleto_discriminator(self):
        if self.esqueleto_D: 
            return self.esqueleto_D
        
        
        self.esqueleto_D = Sequential()

        self.esqueleto_D.add(Dense(2, input_dim=self.var_y, activation='relu', kernel_initializer='he_normal'))
        self.esqueleto_D.add(Dropout(0.5))  
        self.esqueleto_D.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        self.esqueleto_D.add(Dense(1, activation='sigmoid'))
        
        return self.esqueleto_D

    #Esqueleto Gerador
        #Ele vai gerar um dados baseados em dados de entrada
    def esqueleto_generator(self):
        if self.esqueleto_G:
            return self.esqueleto_G
        
        self.esqueleto_G = Sequential()
        
        self.esqueleto_G.add(LSTM(64, input_shape=(self.look_back, self.var), dropout=0.2, recurrent_dropout=0.2))
        self.esqueleto_G.add(Dense(32, activation='relu', kernel_regularizer='l2'))
        self.esqueleto_G.add(Dropout(0.5)) 
        
        self.esqueleto_G.add(Dense(self.var_y, activation='linear')) 

        return self.esqueleto_G

    #Criando o modelo discriminator
    def modelo_discriminator(self): 
        if self.modelo_D:
            return self.modelo_D
        optimizer = Adam(lr=0.0002)
        self.modelo_D = Sequential()
        self.modelo_D.add(self.esqueleto_discriminator())
        self.modelo_D.compile(loss='binary_crossentropy', optimizer=optimizer)
        return self.modelo_D 

    #Criando o modelo adversarial
        #Faz a junção dos dois esqueletos
    def modelo_adversarial(self): 
        if self.modelo_A:
            return self.modelo_A
        optimizer = RMSprop(lr=0.0001)
        self.modelo_A = Sequential()
        self.modelo_A.add(self.esqueleto_generator())
        self.modelo_A.add(self.esqueleto_discriminator())
        self.modelo_A.compile(loss='mean_squared_error', optimizer=optimizer)
        return self.modelo_A
    



class DGAN(object):
    def __init__(self, x_treino=None, look_back = 3, var = None, var_y= None):
        self.look_back = look_back

        self.loss_dis = []
        self.loss_adv = []

        self.x_treino = x_treino[:,3] # Escolhendo a variavel que vai ser prevista
        self.var = var
        self.var_y = var_y

        #Instanciando os modelos
        self.modelos_DGAN = Modelos_DGAN(look_back, var, var_y)
        self.discriminator =  self.modelos_DGAN.modelo_discriminator()
        self.adversarial = self.modelos_DGAN.modelo_adversarial()
        self.generator = self.modelos_DGAN.esqueleto_generator()
        
    #Ativando e desativando o treinamento do discriminator
    def discriminator_trainable(self, val): 
        self.discriminator.trainable = val
        for l in self.discriminator.layers:
            l.trainable = val
    
    #Criando window para entrada de dados
    def create_dataset(self, dataset, look_back=1):
        dataX = []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), :]
            dataX.append(a)
        return np.array(dataX)

    
    #Função para treino
    def train(self, train_steps=5000, batch_size=250):

        for i in range(train_steps):
            
            dados_reais = self.x_treino[np.random.randint(0, self.x_treino.shape[0], size=batch_size),] # Separando dado real para treinamento
            dados_reais = dados_reais.reshape((dados_reais.shape[0],1)) #Colocando os arrays em uma dimensão
            ruidos = np.random.uniform(0, 1.0, size=[batch_size+self.look_back+1, self.var]) #Criando dados de entradas aleatorios
            ruidos = self.create_dataset(ruidos,self.look_back) 
            dados_fake = self.generator.predict(ruidos) #Prevendo dados
            x = np.concatenate((dados_reais, dados_fake))  #Criando o datasetX
            
            
            y = np.ones([2*batch_size, 1])  
            y[batch_size:, :] = 0
            
            #Treinamento do modelo discriminator
            self.discriminator_trainable(True)
            loss_discriminator = self.discriminator.train_on_batch(x, y) 

            #Criando e verificando os dados 
            y = np.ones([batch_size, 1])
            ruidos = np.random.uniform(0, 1.0, size=[batch_size+self.look_back+1, self.var]) 
            ruidos = self.create_dataset(ruidos,self.look_back)
            self.discriminator_trainable(False)
            loss_adversarial = self.adversarial.train_on_batch(ruidos, y)

            self.loss_dis.append(loss_discriminator)
            self.loss_adv.append(loss_adversarial)
            
               
            print('%d: [Discriminator loss: %f]   [Adversarial loss: %f]'  % (i, loss_discriminator, loss_adversarial))
        loss = {}
        loss['discriminator'] = self.loss_dis
        loss['adversarial']   = self.loss_adv
        return loss
            
    
