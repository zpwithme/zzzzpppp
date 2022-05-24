from fcntl import FD_CLOEXEC
from cv2 import CV_32F, sort
import numpy as np
import os
import argparse
from pygame import K_KP_1
import yaml
from pathlib import Path
from tqdm import tqdm
import settings
import cv2
import experiment
import h5py
from superpoint import datasets
from superpoint.settings import DATA_PATH, EXPER_PATH
import tensorflow as tf
from match_features_demo import extract_superpoint_keypoints_and_descriptors

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type = str)
    parser.add_argument("image_path",type = str)
    parser.add_argument("task",type = str)
    parser.add_argument("--export_name",type = str, default  = None)
    args = parser.parse_args()
    

    with open(args.config,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        
    output_dir = Path(EXPER_PATH,'/{}/'.format(args.export_name))
    # if not output_dir.exists():
    #     os.mkdir(output_dir)
        
    # if 'checkpoint' in config:
    #     checkpoint = Path()
    # base_path = Path(DATA_PATH,'COCO/'+args.task+'2014/')
    # pure_image = os.listdir(base_path) 
    # image_path =[]
    # for item in pure_image:
    #     image_path.append(Path(base_path,item))
    
    
    globs = ['*.jpg', '*.png','*.jpeg','*.PNG']
    path = []
    for g in globs: 
        path += list(Path(args.image_path).glob('**/'+g))
        
    paths = sorted(list(set(path)))   
    names = [i.relative_to(args.image_path).as_posix() for i in paths]    
    
    
    
    pred ={}  
    size_new = config['data']['preprocessing']
    
    def process_image(Path,size_new):
        path =str(Path)
       
       
        input_image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        H,W = input_image.shape
        
        size_ = input_image.shape[:2][::-1]
        
        # size_new=tuple(x for x in size_new)                
        image = cv2.resize(input_image,(640,480),interpolation=cv2.INTER_AREA)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, -1)
        return image , size_
        
    # with experiment._init_graph(config, with_dataset=False) as net:
        
    #     net.load(str(checkpoint))
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   '/home/zp/Program/SuperPoint/Dataset/sp_v6')
        
        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        for image,name  in zip(paths,names):
            input_image,size_origin = process_image(image,size_new)
            # print(input_image.shape)
            out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(input_image, 0)})
            
            keypoint_map1 = np.squeeze(out1[0])
            # print(keypoint_map1.shape)
            descriptor_map1 = np.squeeze(out1[1])
            # print(descriptor_map1.shape)
            kp1, desc1_ = extract_superpoint_keypoints_and_descriptors(
                keypoint_map1, descriptor_map1, 5000)
            desc1 = desc1_.transpose(1,0)
            pts=[]
            for kp  in kp1:
              pt = [kp.pt[0],kp.pt[1]]
              pts.append(pt)
            
            scr = np.ones((len(pts),1))
           
            kpts = np.array(pts)
            pred = {
                'keypoints': kpts,
                'descriptors': desc1,
                'scores': scr,
                'image_size': size_origin
             }
            print(pred)
            if 'keypoints' in pred:
                size = np.array([640,480])
                scale = size_origin / size
                pred['keypoints'] = (pred['keypoints']) *scale[None]
                
            with h5py.File('/home/zp/Program/Hierarchical-Localization/outputs/aachen/feats-superpoint-n4096-r1024.h5','a') as fd:
                    if name in fd:
                        del fd[name]
                    grp = fd.create_group(name)
                    for k,v in pred.items():
                        grp.create_dataset(k,data=v)    
                
                
            
            
            
            # print(len(kp1))
            # print(desc1.transpose(1,0).shape)
        
            
            
            
        
        
        
        
    
    
    