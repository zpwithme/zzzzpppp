import  tensorflow as tf
from    tensorflow import keras
import tensorflow.keras.backend as K
from    tensorflow.keras import layers, models, Sequential, backend
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate, Lambda, Input, ZeroPadding2D, AveragePooling2D, DepthwiseConv2D, Reshape

def relu6(x):
    return K.relu(x, max_value=6)

# 保证特征层数为8的倍数
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor*divisor) #//向下取整，除
    if new_v<0.9*v:
        new_v +=divisor
    return new_v

def pad_size(inputs, kernel_size):

    input_size = inputs.shape[1:3]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1,1)

    else:
        adjust = (1- input_size[0]%2, 1-input_size[1]%2)
    
    correct = (kernel_size[0]//2, kernel_size[1]//2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def conv_block (x, nb_filter, kernel=(1,1), stride=(1,1), name=None):

    x = Conv2D(nb_filter, kernel, strides=stride, padding='same', use_bias=False, name=name+'_expand')(x)
    x = BatchNormalization(axis=3, name=name+'_expand_BN')(x)
    x = Activation(relu6, name=name+'_expand_relu')(x)

    return x


def depthwise_res_block(x, nb_filter, kernel, stride, t, alpha, resdiual=False, name=None):

    input_tensor=x
    exp_channels= x.shape[-1]*t  #扩展维度
    alpha_channels = int(nb_filter*alpha)     #压缩维度

    x = conv_block(x, exp_channels, (1,1), (1,1), name=name)

    if stride[0]==2:
        x = ZeroPadding2D(padding=pad_size(x, 3), name=name+'_pad')(x)

    x = DepthwiseConv2D(kernel, padding='same' if stride[0]==1 else 'valid', strides=stride, depth_multiplier=1, use_bias=False, name=name+'_depthwise')(x)

    x = BatchNormalization(axis=3, name=name+'_depthwise_BN')(x)
    x = Activation(relu6, name=name+'_depthwise_relu')(x)

    x = Conv2D(alpha_channels, (1,1), padding='same', use_bias=False, strides=(1,1), name=name+'_project')(x)
    x = BatchNormalization(axis=3, name=name+'_project_BN')(x)

    if resdiual:
        x = layers.add([x, input_tensor], name=name+'_add')

    return x

def MovblieNetV2 (nb_classes, alpha=1., dropout=0):

    img_input = Input(shape=(224, 224, 3))

    first_filter = make_divisible(32*alpha, 8)
    
    x = ZeroPadding2D(padding=pad_size(img_input, 3), name='Conv1_pad')(img_input)
    x = Conv2D(first_filter, (3,3), strides=(2,2), padding='valid', use_bias=False, name='Conv1')(x)
    x = BatchNormalization(axis=3, name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)

    x = DepthwiseConv2D((3,3), padding='same', strides=(1,1), depth_multiplier=1, use_bias=False, name='expanded_conv_depthwise')(x)
    x = BatchNormalization(axis=3, name='expanded_conv_depthwise_BN')(x)
    x = Activation(relu6, name='expanded_conv_depthwise_relu')(x)

    x = Conv2D(16, (1,1), padding='same', use_bias=False, strides=(1,1), name='expanded_conv_project')(x)
    x = BatchNormalization(axis=3, name='expanded_conv_project_BN')(x)

    x = depthwise_res_block(x, 24, (3,3), (2,2), 6, alpha, resdiual=False, name='block_1') 

    x = depthwise_res_block(x, 24, (3,3), (1,1), 6, alpha, resdiual=True, name='block_2') 

    x = depthwise_res_block(x, 32, (3,3), (2,2), 6, alpha, resdiual=False, name='block_3')

    x = depthwise_res_block(x, 32, (3,3), (1,1), 6, alpha, resdiual=True, name='block_4')

    x = depthwise_res_block(x, 32, (3,3), (1,1), 6, alpha, resdiual=True, name='block_5')

    x = depthwise_res_block(x, 64, (3,3), (2,2), 6, alpha, resdiual=False, name='block_6')
    
    x = depthwise_res_block(x, 64, (3,3), (1,1), 6, alpha, resdiual=True, name='block_7')

    x = depthwise_res_block(x, 64, (3,3), (1,1), 6, alpha, resdiual=True, name='block_8')

    x = depthwise_res_block(x, 64, (3,3), (1,1), 6, alpha, resdiual=True, name='block_9')

    x = depthwise_res_block(x, 96, (3,3), (1,1), 6, alpha, resdiual=False, name='block_10')

    x = depthwise_res_block(x, 96, (3,3), (1,1), 6, alpha, resdiual=True, name='block_11')

    x = depthwise_res_block(x, 96, (3,3), (1,1), 6, alpha, resdiual=True, name='block_12')

    x = depthwise_res_block(x, 160, (3,3), (2,2), 6, alpha, resdiual=False, name='block_13')

    x = depthwise_res_block(x, 160, (3,3), (1,1), 6, alpha, resdiual=True, name='block_14')

    x = depthwise_res_block(x, 160, (3,3), (1,1), 6, alpha, resdiual=True, name='block_15')

    x = depthwise_res_block(x, 320, (3,3), (1,1), 6, alpha, resdiual=False, name='block_16')

    if alpha >1.0:
        last_filter = make_divisible(1280*alpha,8)
    else:
        last_filter=1280

    x = Conv2D(last_filter, (1,1), strides=(1,1), use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(axis=3, name='Conv_1_bn')(x)
    x = Activation(relu6, name='out_relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax',use_bias=True, name='Logits')(x)

    model = models.Model(img_input, x, name='MobileNetV2')

    return model

def prepocess(x):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [224,224])
    x =tf.expand_dims(x, 0) # 扩维
    x = preprocess_input(x)
    return x


#preprocess_input 等效于/255  -0.5  *2（缩放到-1 ，1）
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


def main():

    model = MovblieNetV2(1000, 1.0, 0.2)
    model.summary()

    weight_path=r'C:\Users\jackey\.keras\models\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'

    if os.path.exists(weight_path):
        model.load_weights(weight_path, )

    img=prepocess('Car.jpg')

    Predictions = model.predict(img)
    print('Predicted:', decode_predictions(Predictions, top=3)[0])


if __name__=='__main__':
    main()

