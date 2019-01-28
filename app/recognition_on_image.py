import numpy as np
import os
from keras.preprocessing import image
from keras.models import model_from_json


def load_and_compile():
    json_file = open("bin_class_net.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    # Создаем модель
    model = model_from_json(loaded_model_json)
    # Загружаем сохраненные веса в модель
    model.load_weights("bin_class_net.h5")
    print("Загрузка сети завершена")
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
  
    
def predict_class(model, img_path):
    
    
    img = image.load_img(img_path, target_size=(150, 150))
    
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    
    return prediction

def main():
    dir_name = input('Введите дирректорию, в которой находится фаил:  ')
    file_name = input('Введите имя файла с раширением:  ')
    
    img_path = file_name = os.path.join(dir_name, file_name)

    model = load_and_compile()
    prediction = predict_class(model, img_path)
    if prediction > 0.5:
        print('human')
    else:
        print('car')
        
if __name__ == '__main__':
    main()
