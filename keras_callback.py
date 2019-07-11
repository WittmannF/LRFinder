from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

class LRFinder(Callback):
    def __init__(self, min_lr, max_lr, mom=0.9, stop_multiplier=None):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        if stop_multiplier is None:
            self.stop_multiplier = -20*self.mom/3 + 10 # 4 if mom=0.9, 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier
        
    def on_train_begin(self, logs={}):
        n_iterations = self.params['samples']//self.params['batch_size']
        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, \
                                           num=n_iterations+1)
        self.losses=[]
        self.iteration=0
        self.best_loss=0
        
    
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        
        if self.iteration!=0: # Make loss smoother using momentum
            loss = self.losses[-1]*self.mom+loss*(1-self.mom)

        if self.iteration==0 or loss < self.best_loss: 
            self.best_loss = loss
            
        lr = self.learning_rates[self.iteration]
        K.set_value(self.model.optimizer.lr, lr)
        
        self.losses.append(loss)            
        self.iteration += 1
        if loss > self.best_loss*self.stop_multiplier: # Stop criteria
            self.model.stop_training = True
    
    def on_train_end(self, logs=None):
        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()
