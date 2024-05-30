"""
builds and compiles a convolutional neural network (CNN) using TensorFlow's Keras API.
This CNN is designed for image classification, specifically to classify images into one
of three classes (rock, paper, or scissors).
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from extract_data import file_name, target_dir, data_set
from generator import train_generator, validation_generator

# build model
# Sequential: This is a linear stack of layers in Keras.
def build_rpc_model():
    print("building RPC model \n")
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),# This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
    MaxPooling2D((2, 2)),# This layer performs max pooling operations to reduce the spatial dimensions of the input.
    Flatten(),# This layer flattens the input, transforming the 2D matrix into a 1D vector.
    Dense(512, activation='relu'),# This layer is a fully connected neural network layer.
    Dropout(0.5),# This layer is used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
    Dense(3, activation='softmax')  # Ensure this is 3 for three classes
    ])

    # compile the model
    model.compile(
        optimizer='adam',# The Adam optimizer is used to minimize the loss function.
        loss='categorical_crossentropy',# The categorical crossentropy loss function is used for multi-class classification problems where the labels are one-hot encoded.
        metrics=['accuracy'] # The accuracy metric is used to evaluate the performance of the model.
        )
    
    return model

#train model
def train_rpc_model(train_generator, validation_generator):
    data_set(file_name, target_dir)
    model = build_rpc_model()
    print("training RPC model \n")
    history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
    )
    
    # evaluate model
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation accuracy: {accuracy*100:.2f}%')
    
    # save model
    save_rpc_model(model)

# save model
def save_rpc_model(model):
    print("saving RPC model \n")
    # Save the model in HDF5 format
    file_name = "RPSClassifier.h5"
    model.save(file_name)
    # Download the model file
    # files.download(file_name) #for colab only
    

train_rpc_model(train_generator, validation_generator)