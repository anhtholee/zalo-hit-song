# This code is a fork from https://github.com/dkn22/embedder with some modification
from .base import Base
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, Model, model_from_json
from keras.layers import (Dense, Dropout, Embedding,
                          Activation, Input, concatenate, Reshape, Flatten, GaussianNoise, GaussianDropout, LeakyReLU)
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, TFOptimizer
from keras import backend as K
from keras.models import load_model
from keras import regularizers
from .metrics import r2, rmse_k
# from .COCOB import COCOB
# from tensorflow.python.keras.optimizers import TFOptimizer

def swish(x):
    return K.sigmoid(x) * x

class Embedder(Base):

    def __init__(self, emb_sizes, model_json=None, weight_path=None, loss='mean_squared_error', hiddens=None, dropout=None, activation=None):

        super(Embedder, self).__init__(emb_sizes, model_json, weight_path, loss, hiddens, dropout, activation)

    def fit(self, X, y,
            batch_size=256, epochs=100,
            checkpoint=None,
            early_stop=None, verbose=False,):
        '''
        Fit a neural network on the data.

        :param X: input DataFrame
        :param y: input Series
        :param batch_size: size of mini-batch
        :param epochs: number of epochs for training
        :param checkpoint: optional Checkpoint object
        :param early_stop: optional EarlyStopping object
        :return: Embedder instance
        '''
        nnet = self._create_model(X, model_json=self.model_json)
        if self.weight_path is not None:
            nnet.load_weights(self.weight_path)
            self.model = nnet
            return self
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # opt = TFOptimizer(COCOB())
        rlr = ReduceLROnPlateau(monitor='val_rmse_k', patience=2, verbose=1, factor=0.3, min_lr=0.00001, epsilon=0.001)
        nnet.compile(loss=self.loss,
                     optimizer=opt,
                     metrics=[rmse_k])

        callbacks = list(filter(None, [checkpoint, early_stop, rlr]))
        callbacks = callbacks if callbacks else None

        x_inputs_list = self._prepare_inputs(X)

        nnet.fit(x_inputs_list, y.values, batch_size=batch_size,
                 epochs=epochs,
                 callbacks=callbacks,
                 validation_split=0.12, 
                 shuffle=True, 
                 verbose=verbose)
        self.model = nnet

        return self

    def save_weights(self, weight_path):
        self.model.save_weights(weight_path)

    def fit_transform(self, X, y,
                      batch_size=256, epochs=100,
                      checkpoint=None,
                      early_stop=None,
                      as_df=False
                      ):
        '''
        Fit a neural network and transform the data.

        :param X: input DataFrame
        :param y: input Series
        :param batch_size: size of mini-batch
        :param epochs: number of epochs for training
        :param checkpoint: optional Checkpoint object
        :param early_stop: optional EarlyStopping object
        :return: transformed data
        '''
        self.fit(X, y, batch_size, epochs,
                 checkpoint, early_stop)

        return self.transform(X, as_df=as_df)

    def rapport(self):
        return self.model.summary()

    def _default_nnet(self, X):

        emb_sz = self.emb_sizes
        numerical_vars = [x for x in X.columns
                          if x not in self._categorical_vars]

        inputs = []
        flatten_layers = []

        for var, sz in emb_sz.items():
            input_c = Input(shape=(1,), dtype='int32')
            embed_c = Embedding(*sz, input_length=1)(input_c)
            # embed_c = Dropout(0.25)(embed_c)
            flatten_c = Flatten()(embed_c)

            inputs.append(input_c)
            flatten_layers.append(flatten_c)

        input_num = Input(shape=(len(numerical_vars),), dtype='float32')
        flatten_layers.append(input_num)
        inputs.append(input_num)

        flatten = concatenate(flatten_layers, axis=-1)
        hiddens = self.hiddens if self.hiddens is not None else [512, 128]
        activation = self.activation if self.activation is not None else 'relu'

        fc1 = Dense(hiddens[0], kernel_initializer='random_uniform',)(flatten)
        # fc1 = GaussianNoise(0.01)(fc1)
        fc1 = BatchNormalization()(fc1)
        if activation == 'swish':
            fc1 = Activation(swish)(fc1)
        elif activation == 'lrelu':
            fc1 = LeakyReLU(alpha=0.2)(fc1)
        else:
            fc1 = Activation(activation)(fc1)
        if self.dropout is not None:
            fc1 = Dropout(self.dropout)(fc1)

        fc2 = Dense(hiddens[1], kernel_initializer='random_uniform',)(fc1)
        # fc2 = GaussianNoise(0.01)(fc2)
        fc2 = BatchNormalization()(fc2)
        if activation == 'swish':
            fc2 = Activation(swish)(fc2)
        elif activation == 'lrelu':
            fc2 = LeakyReLU(alpha=0.2)(fc2)
        else:
            fc2 = Activation(activation)(fc2)
        # fc2 = Dropout(0.2)(fc2)

        output = Dense(1, kernel_initializer='random_uniform')(fc2)
        nnet = Model(inputs=inputs, outputs=output)

        return nnet
