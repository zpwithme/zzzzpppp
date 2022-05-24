from audioop import lin2adpcm
from os import name
from numpy.lib.arraypad import pad
from superpoint.models.backbones.Mobile_netV2 import conv_block
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.python.keras.backend import shape

import tensorflow as tf
from tensorflow import layers as tfl



# class Mobilenet_V2():
#     def __init__(self, *,inp_shape = (224,224,3), rho =1.0, alpha = 1.0 , expansion = 6.0 ,classes = 2,dropout = 0.0):
#         assert alpha >0 and alpha <=1
#         assert rho >0 and rho <=1
#         self._inp_shape = inp_shape
#         self._rho = rho
#         self._alpha = alpha
#         self._expansion = expansion
#         self._classes = classes
#         self._droppout = dropout
        
        
def Conv_Block(input,nb_filter, name,data_format='channels_first',kernel=(1,1),stride=(1,1),
                   batch_normalization=True, kernel_reg=0.,linear = False, **params):
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            print(input.shape)
            print(nb_filter)
            x= tfl.conv2d(input,nb_filter,kernel,name= name+'Conv',strides = stride,padding = 'same',data_format='channels_first',use_bias = False)
            x= tfl.batch_normalization(x,name=name+'_BN',fused=True,axis=1)
            if linear == False:
              x= tf.nn.relu6(x,name=name+'_relu')
              print(x.shape)
             
        return x     
            
   
# def depthwiseconv(x, strides_depthwise,  t,  name,resdiual=False):
#         with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
#             tf.nn.depthwise_conv2d(x,kernel=(3,3),stride=strides_depthwise,
#                                    padding='same' if strides_depthwise ==1 else 'valid',name = name +'_deepwise')   
    
def depthwise_res_block(input,strides_depthwise,output_channels,expansion,name,data_format='channels_first',training=True,residual = False,**params):
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            print(strides_depthwise)
            filter = int(input.shape[1])
            print(type(filter))
            
            exp_channels= int(filter) * expansion
            print(expansion)

            x = Conv_Block(input, exp_channels,name= name +'_pad',data_format='channels_first',**params)
            print(x.shape)
            
            filter_ = tf.get_variable("filter1", [3, 3, exp_channels,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            if strides_depthwise == 2:
              x = tf.nn.depthwise_conv2d(x,filter_,padding='SAME' , strides=[1,1,2,2],
                                       data_format='NCHW',name = name+'_depthwise')
            else:   x = tf.nn.depthwise_conv2d(x,filter_,padding='SAME' , strides=[1,1,1,1],
                                       data_format='NCHW',name = name+'_depthwise')
            print(x.shape)
            x = Conv_Block(x,output_channels,name =name +'_project',data_format='channels_first',**params)
            
            if strides_depthwise ==1 and output_channels == filter:
                x = x +input 
                
        return x        
            
            
            
def bottleneck_block_(input,s,c,t,n,name):
    """
      s: stride
      c: channel
      t: expansion factor
      n: repeat time
    """
    
    x= depthwise_res_block(input,s,c,t,name = name+'0')
    print(name)
    print(x.shape)
    print(type(x))
    for i in range(n-1):
        x= depthwise_res_block(x,1,c,t,name = name +str(i+1))
    return x     
    
def mobile_NetV2_backbone(inputs, **config):
    
    with tf.variable_scope('MobileNetV2',reuse = tf.AUTO_REUSE):
        x = tfl.conv2d(inputs, filters=32, kernel_size=(3,3), name='conv',
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(0.),strides=(2,2),
                       data_format='channels_first', padding = 'SAME')
  
  
        x= bottleneck_block_(x, s =1,c= 16,t =1, n =1,name = 'block_1')
        x= bottleneck_block_(x, s =2,c= 24,t = 6 ,n =2,name = 'block_2')
        
        x= bottleneck_block_(x, s= 2,c= 32, t = 6, n=3,name = 'block_3')
    
    
    return x
    
        
    def _depthwiseconv(x, strides_depthwise,  t,  name,resdiual=False):
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            tf.nn.depthwise_conv2d(x,kernel=(3,3),stride=strides_depthwise,padding='same' if strides_depthwise ==1 else 'valid',name = name +'_deepwise')
        
        
        
        
        # return models.Sequential([
        #     DepthwiseConv2D(kernel_size = (3,3), strides = stride, padding = 'same' if stride ==1 else 'valid', use_bias = False),
        #     BatchNormalization(),
        #     ReLU(max_value = 6.)
        # ])
        
        
        
        
# def pointwiseconv(self, *,filters: int, Linear: bool):
#         layer = models.Sequential([
#         Conv2D(filters = int(filters * self._alpha), kernel_size=(1,1), strides = (1,1), padding = 'same',use_bias = False),
#         BatchNormalization(),
#         ])
#         if linear == False:
#             layer.add(ReLU(max_value = 6.))
#         return layer
# def standardconv(self):
#         return models.Sequential([
#             conv2D(filters = 32, kernel_size = (3,3), strides = (2,2), use_bias = False),
#             BatchNormalization(),
#             ReLU(max_value= 6.)
#         ])
# def inverted_residual_block_(self,x, *, strides_depthwise: int ,filter_pointwise: int ,expansion: int):
#         filter = int(filter_pointwise * self._alpha)
#         fx = self._pointwiseconv(filters=filter*expansion,linear = False)(x)
#         fx = self._depthwiseconv(strides= strides_depthwise)(fx)
#         fx = self._pointwiseconv(filters = filter, linear = True)(fx)
#         if strides_depthwise == 1 and x.shape[-1] ==filter_pointwise:
#             return add([fx,x])
#         else: 
#             return fx
#     def _bottleneck_block_(self, x ,*, s:int, c: int, t: int, n :int):
#         '''
#             s: strides
#             c: output channels
#             t :expansion factor
#             n :repeat
#         '''    
#         x = self._inverted_residual_block_(x, strides_depthwise=s, filter_pointwise=c, expansion=t)
#         for i in range(n-1):
#             x = self._inverted_residual_block_(x, strides_depthwise= 1 , filter_pointwise= c , expansion=t)
#         return x
#     def build(self):
#         feature_map_H = int (self._rho * self._inp_shape[0])
#         feature_map_W = int (self._rho * self._inp_shape[1])
#         img_inp = Input(shape = (feature_map_H, feature_map_W,1))
        
#         x= self._standardconv()(img_inp)
        
#         x= self._bottleneck_block_(x, s =1,c= 16,t =1, n =1)
        
#         x= self._bottleneck_block_(x, s =2,c= 24,t = self._expansion ,n =2)
        
#         x= self._bottleneck_block_(x, s= 2,c= 32, t = self._expansion, n=3)
        
#         x= self._bottleneck_block_(x, s=2, c =64,t =self._expansion, n=4)
        
#         x= self._bottleneck_block_(x,s =1, c= 96, t=self._expansion,n = 3)
   
#         x = self._bottleneck_block_(x, s=1,c= 160,t =self._expansion, n=3)
        
#         x= self._bottleneck_block_(x, s=1,c =320, t =self._expansion, n=1)
        
#         x= self._pointwiseconv(filters = 1280, linear = False)(x)
        
#         x= GlobalAveragePooling2D()(x)
#         x = Dropout(self._droppout)
#         x = Dense(self._classes, activation='softmax')(x)
#         return models.Model(img_inp, x ,name = 'mobilenetv2')
        
                
        
        
        
        
        
            
    
        
        
        
            
            
        
        