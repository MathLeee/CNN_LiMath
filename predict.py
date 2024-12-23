import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from util import draw_one_sample
def predict(model, image_file):
    model=load_model(model)
    img=Image.open(image_file)
    draw_one_sample(img)
    img=img.convert('L')
    img=np.array(img)
    #img=255-img
    img=img/255.0
    x=img.reshape(1,img.shape[0],img.shape[1], 1)
    preds=model.predict(x)
    print("***************************************")
    print("预测为特定数字的概率")
    print("***************************************")
    for i in range(10):
        print("识别为"+str(i)+"的概率:{0:.2f}".format(preds[0][i]*100.0))
    print("***************************************")
    print("最终AI认为这个图片代表的数字为:"+str(np.argmax(preds[0])))
    print("***************************************")
if __name__ == '__main__':
    predict('model.h5','test/1.png')