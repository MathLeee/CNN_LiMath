from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from util import load_data
from util import draw_training_performance_graph
from le_net_5 import le_net_5


def train(batch_size=256, epochs=20, num_classes=10):
    (x_train, y_train), (x_test, y_test) = load_data('mnist.npz')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    sample_shape = x_train[0].shape
    input_shape = (sample_shape[0], sample_shape[0], 1)
    K.set_image_data_format('channels_last')
    sample_no = x_train.shape[0]
    rows = input_shape[0]
    columns = input_shape[1]
    channels = input_shape[2]
    x_train = x_train.reshape(x_train.shape[0], rows, columns, channels)
    x_test = x_test.reshape(x_test.shape[0], rows, columns, channels)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    model = le_net_5(shape=input_shape)
    print(model.summary())
    model.compile(loss=categorical_crossentropy,
                  optimizer=AdamW(learning_rate=0.001, weight_decay=0.0001),#AdamW(learning_rate=0.001, weight_decay=0.0001),#SGD(learning_rate=0.01, momentum=0.9, nesterov=True),#Adam(learning_rate=0.001)
                  metrics=['accuracy'])
    # 训练模型
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        verbose=2,
                        validation_split=0.1)
    # 绘制训练性能图
    draw_training_performance_graph(history,epochs)
    # 保存模型
    model.save('model.h5')
    # 用测试集评估模型
    predict = model.evaluate(x_test, y_test, batch_size=batch_size)
    print()
    print("测试集损失值="+str(predict[0]))
    print("测试集正确率=" + str(predict[1]))

#运行训练
if __name__ == '__main__':
    train()