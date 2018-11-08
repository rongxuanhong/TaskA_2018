import numpy as np
import tensorflow as tf


class MixupGenerator:
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True):
        self.X_train = X_train,
        self.y_train = y_train,
        self.batch_size = batch_size,
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)

    def __shuffle_indexs(self):
        indexs = np.arange(self.sample_num)
        if self.shuffle:
            indexs = np.random.permutation(self.sample_num)
        return indexs

    def __data_generation(self, batch_indexs):
        pass

    def __call__(self, *args, **kwargs):
        while True:
            indexs = self.__shuffle_indexs()

