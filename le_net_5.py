from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model,Sequential

def le_net_5(shape):
    # 输入层,28x28x1
    inputs = Input(shape=shape)
    # 第一层:卷积层,6核5x5卷积,步长1
    x = Conv2D(6, kernel_size=(5, 5), activation='relu', padding='same')(inputs)
    # 池化层,2x2,步长2x2
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # 第二层:卷积层,16核5x5卷积,步长1
    x = Conv2D(16, kernel_size=(5, 5), activation='relu', padding='valid')(x)
    # 池化层,2x2,步长2x2
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # 随机失活层(25%的神经元失活)
    x = Dropout(0.25)(x)
    # 扁平层,多维变一维
    x = Flatten()(x)
    # 第三层:全连接层
    x = Dense(120, activation='relu')(x)
    # 第四层:全连接层
    x = Dense(84, activation='relu')(x)
    # 随机失活层(50%的神经元失活)
    x = Dropout(0.50)(x)
    # 第五层:输出层
    outputs = Dense(10, activation='softmax')(x)

    # 创建卷积神将网络模型
    model = Model(inputs=inputs, outputs=outputs)
    return model

