import sys
from keras.backend import sigmoid
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


#Definicion de modelo CNN
def define_model():
    model = VGG16(include_top=False, input_shape=(224, 224, 3))

    #Bloqueo de capas no entrenables
    for layer in model.layers:
        layer.trainable = False
    
    #Nuevas capas entrenables
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    #Creacion del modelo
    model = Model(inputs = model.inputs, output=output)

    #Creacion del optimizador
    opt = SGD(lr=0.001, momentum=0.9)

    #Compilacion del modelo
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(history):
    
    #Loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')

    #Accuracy
    pyplot.subplot(211)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

    #Mostrar plot / Cambiar por archivo eventualmente
    pyplot.show()



def run_test():

    model = define_model()

    datagen = ImageDataGenerator(featurewise_center=True)

    datagen.mean = [123.68, 116.779, 103.939]

    #Carga de datos, modificar el path!!!!!
    train_it = datagen.flow_from_directory('', class_mode='binary', batch_size=64, target_size=(224, 224)) #SET PATH FROM COLAB
    test_it = datagen.flow_from_directory('', class_mode='binary', batch_size=64, target_size=(224, 224)) #SET PATH FROM COLAB

    #FIT
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
    validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)


    #EVAL
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    summarize_diagnostics(history)

run_test()