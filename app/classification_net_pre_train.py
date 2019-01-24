"""
приложение для обучения сети бинарной классификации изображений.
при вызове необходимо ввести путь к данным для обучения. 
после выполнения программы в рабочей дирректории создаются файлы с конфигурацией сети.
и весами сети.
В процессе выполнения программа выводит данные о колличестве изображения для обучения сети,
о структуре сети и колличестве параметров сети, а так же график обучения сети.
@author: марк
"""

# Импорты библиотек
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def load_pre_trained_model(name_model, img_width, img_height, ):
    """ Функция загружает модель классификации натреннированной на датасете
    imagenet запрещается обучение весов сети , печатается колличество 
    парраметров сети"""
    
    
    pre_train_net = name_model(include_top=False,
                               weights='imagenet', 
                               input_shape=(img_width, img_height, 3))
    pre_train_net.trainable = False
    
    return pre_train_net


def  additional_net(pre_train_net):
    """ Функция создает итоговую сеть, первой частью которой является 
    сеть, переданная в функцию, а затем идут два полносвязных слоя для 
    классификации.
    На выход функция отдает скомпилированную модель"""
    
    
    model = Sequential()
    model.add(pre_train_net)
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5), 
              metrics=['accuracy'])
    model.summary()
    
    return model


def train_gen_and_fit(train_dir, val_dir, test_dir, img_width, img_height,
                      batch_size, model, nb_train_samples,
                      nb_validation_samples, nb_test_samples):
    """ Функция сбора изображений для обучения сети, обучения сети и оценке
    качества обучения. 
    На вход принимает дирректории с файлами, размер изображений,
    размер мини выборки, скомпилированную модель, количество тренировочных
    файлов, колличество проверочных файлов, колличество тестовых файлов.
    На выход возвращает обученную модель"""
    
    
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')  
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=3,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)
    
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))
    return model, history


def save_model(model):
    """ Функция преобразует структуру модели к формату JSON 
    Сохраняет модель и веса модели в рабочей дирректории"""

    model_json = model.to_json()
    json_file = open("bin_class_net.json", "w")
    json_file.write(model_json)
    json_file.close()
    model.save_weights("bin_class_net.h5")

def vis_train(history):
    """ Функция визуализации процесса обучения, помогает понять качество
    обученной модели.
    На вход принимает обученную модель.
    Строит график точности обучения"""
    
    
    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color = 'r', label = 'Train')
    plt.show()
    

def handler(dir_name = 'C:\\Users\\марк\\Documents\\classification_net'):
    """ функция инициализации переменных и порядка запуска обучения.
    На вход принимает дирректорию с данными для обучения.
    Инициализирует колличественные переменные, дирректории подкаталогов.
    """
    
    
    # Каталог с данными для обучения
    train_dir = os.path.join(dir_name, 'train')
    # Каталог с данными для проверки
    val_dir = os.path.join(dir_name, 'val')
    # Каталог с данными для тестирования
    test_dir = os.path.join(dir_name, 'test')
    # Размеры изображения
    img_width, img_height = 150, 150
    # Размер мини-выборки
    batch_size = 5
    # Количество изображений для обучения
    nb_train_samples = 200
    # Количество изображений для проверки
    nb_validation_samples = 100
    # Количество изображений для тестирования
    nb_test_samples = 100
#    Имя модели для переноса обучения
    name_model = VGG16
    pre_train_net = load_pre_trained_model(name_model, img_width, img_height)
    model = additional_net(pre_train_net)
    model, history = train_gen_and_fit(train_dir, val_dir, test_dir, img_width, 
                              img_height, batch_size, model, nb_train_samples,
                              nb_validation_samples, nb_test_samples)
    save_model(model)
    vis_train(history)


def main():
    """ Функция запуска работы программы.
    при выполнении получает от пользователя дирректорию с данными и запускает
    выполнение функции инициализации переменных и порядка запуска обучения """
    
    
    dir_name = str(input('введите дирректорию с данными(/test, /train, /val)'))
    handler(dir_name)
    
if __name__ == '__main__':
    main()
