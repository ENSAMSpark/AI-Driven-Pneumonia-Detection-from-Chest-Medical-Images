"""
Construction du modèle CNN
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Flatten, Dense, 
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from config import *

class PneumoniaCNN:
    def __init__(self, input_shape=INPUT_SHAPE, config=MODEL_CONFIG):
        self.input_shape = input_shape
        self.config = config
        self.model = None
        
    def build_model(self):
        """
        Construit le modèle CNN
        """
        model = Sequential()
        
        # Couche d'entrée
        model.add(Input(shape=self.input_shape))
        
        # Couches convolutionnelles
        for i, layer_config in enumerate(self.config['conv_layers']):
            model.add(Conv2D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                padding='same',
                activation='relu',
                kernel_regularizer=l2(0.001)  # Régularisation L2
            ))
            
            model.add(BatchNormalization())
            
            if layer_config.get('dropout'):
                model.add(Dropout(layer_config['dropout']))
            
            # Ajouter MaxPooling après chaque 2 couches conv
            if i % 2 == 1 or i == len(self.config['conv_layers']) - 1:
                model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        
        # Flatten
        model.add(Flatten())
        
        # Couches denses
        for units in self.config['dense_layers']:
            model.add(Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l2(0.001)
            ))
            model.add(Dropout(0.5))  # Dropout plus agressif pour les couches denses
            model.add(BatchNormalization())
        
        # Couche de sortie
        model.add(Dense(
            units=1,
            activation=self.config['output_activation']
        ))
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=LEARNING_RATE):
        """
        Compile le modèle avec l'optimiseur et la fonction de perte
        """
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                'Precision',
                'Recall',
                'AUC'
            ]
        )
        
        return self.model
    
    def summary(self):
        """
        Affiche le résumé du modèle
        """
        if self.model:
            return self.model.summary()