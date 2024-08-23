from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, ReLU, Input
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Use Input layer as the first layer
    model.add(Conv1D(filters=16, kernel_size=3, padding='same'))
    model.add(ReLU())

    model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
    model.add(ReLU())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

    return model

def compile_model(model, learning_rate=0.001):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model