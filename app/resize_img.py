# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 17:52:55 2019

@author: марк
"""
import os
from PIL import Image

def resize(dir_image): 
    img = Image.open(dir_image)
    width = 150
    height = 150
    resized_img = img.resize((width, height), Image.ANTIALIAS)
    resized_img.save(dir_image)
    
def worker(dir_name):
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.jpg':
            resize(dir_name + fname)

def main():
    worker('C:\\Users\\марк\\Documents\\classification_net\\test\car\\')
    
if __name__ == '__main__':
    main()