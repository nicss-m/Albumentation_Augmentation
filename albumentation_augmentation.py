# required modules
import os
import cv2
import time
import json
import queue
import base64
import random
import threading
import numpy as np
import albumentations as A
from shutil import copyfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

class album_augmentation:
    
    def __init__(
        self,
        check_by = "",
        transform = None,
        type_of_adjust = 0,
        child_functions = ['get_transformation','set_transformation',' compute_scale_adjustment'],
        aan = ['polygon-json','yolo-txt','pascal_voc-xml','bbox-json'],
        acc_img_format = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    ):
        super(album_augmentation).__init__()
        self.check_by = check_by
        self.transform = transform
        self.type_of_adjust = type_of_adjust
        self.child_functions = child_functions
        self.aan = aan
        self.acc_img_format = acc_img_format

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)

    def exec_time(self, stop, percent_queue):
        strtime = datetime.now()
        starttime = strtime.strftime('%H:%M:%S')
        print("Time Start:",starttime)
        count = 0
        prev_sec = 0
        percent = 0.1
        sign_percent = ""
        sign_count = 0
        eta = timedelta(seconds=0)
        while True:
            loading = ['\\\\\\\\\\\\\\\\\\\\','||||||||||','//////////','----------']
            curtime = datetime.now()
            elapsed = curtime-strtime
            elapsedtime = str(elapsed).split('.', 2)[0]
            time.sleep(0)
            try:
                percent = percent_queue.get(timeout=1)
            except queue.Empty:
                pass

            x = percent/10
            if x>1:
                x = int(x)
                sign_percent = "#"*x
                sign_count = x

            print("Time Elapsed: {}\t\tWorking on it :  {} {}% | ETA: {}".format(elapsedtime,sign_percent+loading[count][sign_count:10],percent,
                                                                                     str(eta).split('.', 2)[0]), end='\r')
            if elapsed.seconds!=prev_sec:
                count+=1
                prev_sec+=1
                if elapsed.seconds%5==0:
                    eta = timedelta(seconds=((100/percent)*elapsed.total_seconds())-elapsed.total_seconds())
                if eta.seconds>0:
                    eta-=timedelta(seconds=1)

            if count>3:
                count=0
            if stop():
                eta=timedelta(seconds=0)
                print("Time Elapsed: {}\t\tWork Done :  {} {}% | ETA: {}".format(elapsedtime,sign_percent+loading[count][sign_count:10],percent,
                                                                                     str(eta).split('.', 2)[0]), end='\r')
                endtime = curtime.strftime('%H:%M:%S')
                print("")
                print("Time End: "+endtime)
                break
  
    def check_inputs(self,func=None,path=None,img_name=None,scale=None,w_limit=None,h_limit=None,new_name=None,accept_img_format=None, w=None, h=None,
                     annotation_format=None,save_path=None,transform=None,save_image_as=None,except_list=None,random_seed=None,n_points_exist=None):
        #### General Function Parameter Checking ####
        # check if given path is valid
        if path!=None:
            if os.path.isdir(path) == False:
                    raise FileNotFoundError("no such directory detected in the specified path. please input a valid folder directory")

        # check and create save path if none given
        if func not in self.child_functions:
            if save_path == None:
                save_path = func
                if os.path.isdir(func) == False:
                    os.makedirs(func)
            else:
                if os.path.isdir(save_path) == False:
                    os.makedirs(save_path)
            save_path = os.path.normpath(save_path)

        # check annotation format
        if annotation_format!=None:
            if type(annotation_format)!=str:
                raise TypeError("invalid annotation format! expected input is of type string")
            else:
                annotation_format = annotation_format.lower()
                if annotation_format not in self.aan:
                    raise ValueError("invalid annotation format! accepted format includes 'pascal_voc-xml','yolo-txt', 'polygon-json' and 'bbox-json'")

        # check image format
        if accept_img_format==None:
            accept_img_format=self.acc_img_format
        else: 
            if type(accept_img_format)==tuple:
                if all(i in self.acc_img_format for i in accept_img_format)==False:
                    raise ValueError("invalid image format in accept_img_format! accepted format includes '.png', '.jpg', '.jpeg', '.tiff', '.bmp', and '.gif'")
            else:
                raise TypeError("invalid input type in accept_img_format! expected input is of type tuple. Note: for single element tuple, add a comma on the end. example usage: ('.jpeg',)")
        if save_image_as!=None:
            if type(save_image_as)==tuple:
                if len(save_image_as)==1:
                    if save_image_as[0] not in self.acc_img_format:
                         raise ValueError("invalid image format in save_image_as! accepted format includes '.png', '.jpg', '.jpeg', '.tiff', '.bmp', and '.gif'")
                else:
                    raise ValueError("more than one image format was specified in save_image_as. please specify only one image format i.e. ('.jpeg',)")
            else:
                raise TypeError("invalid input type in save_image_as! expected input is of type tuple. Note: for single element tuple, add a comma on the end. example usage: ('.jpeg',)")

        #### Specific Function Parameter Checking ####
        if func == 'adjust_img_and_annotate':
            # check scale
            if type(scale)==tuple and len(scale)==2:
                self.type_of_adjust = 3
            elif type(scale)==int or type(scale)==float:
                if scale>3 or scale<=0:
                    raise ValueError("overlapping scale detected! limit is between 0-3")
                else:
                    self.type_of_adjust = 0
            elif type(scale)==str:
                if scale.lower() == "up" or scale.lower() == "down":
                    if w_limit==None and h_limit==None:
                        raise Exception("scale up or down requires at least one limit value, either width or height limit or could be both") 
                    else:
                        if scale.lower() == "up":
                            self.type_of_adjust = 1
                        else:
                            self.type_of_adjust = 2
                else:
                    raise ValueError("invalid string value in scale! expected input is either 'up' or 'down' only")
            else:
                raise TypeError("invalid scale format! expected input are of type int, float, string ('up' or 'down') and tuple for fix resize (i.e. 0.8 or (200,200))")

            # check width and height limit
            if w_limit!=None or h_limit!=None:
                if type(scale)!=str:
                    raise Exception("invalid pass! width and height limit is only applicable during scale 'up' and 'down'")
                if w_limit!=None:
                    if type(w_limit)!=int:
                        raise TypeError("invalid input type in w_limit. expected input is of type int")
                if h_limit!=None:
                    if type(h_limit)!=int:
                        raise TypeError("invalid input type in h_limit. expected input is of type int")

            # check except list
            if except_list==None:
                except_list = []
            else:
                if type(except_list)!=list:
                    raise TypeError("invalid input type in except_list! expected input is of type list")  
                    
            # set check_by to prevent checking repition in child functions
            self.check_by = func
            return path,scale,w_limit,h_limit,save_image_as,except_list,accept_img_format,annotation_format,save_path

        elif func == 'A_augmentation':
            
            # check transform ang get predefined transformation base on annotation format if none given
            if transform==None:
                transform = self.get_transformation(annotation_format, random_seed)
            elif str(type(transform))!="<class 'albumentations.core.composition.Compose'>":
                raise TypeError("invalid transform given! expected input is of type 'albumentations.core.composition.Compose'")

            # check except list
            if except_list==None:
                except_list = []
            else:
                if type(except_list)!=list:
                    raise TypeError("invalid input type in except_list! expected input is of type list")  
                    
            # check random seed
            if random_seed!=None:
                if type(random_seed)!=int:
                    raise TypeError("invalid input type in random_seed! expected input is of type int")
                    
            # check negative points
            if n_points_exist==None:
                n_points_exist==True
            else:
                if type(n_points_exist)!=bool:
                    raise TypeError("invalid input type in n_points_exist! expected input is of type boolean")
            
            # set check_by to prevent checking repition in child functions
            self.check_by = func
            
            return path,transform,annotation_format,save_image_as,except_list,accept_img_format,random_seed,n_points_exist,save_path

        elif func=='polygon_or_bbox_json_resize' or func=='yolo_txt_resize' or func=='pascal_voc_xml_resize' or func=='image_only_resize':
            
            # check image name
            if type(img_name)!=str:
                raise TypeError("invalid input type in img_name! expected input is of type string")
            # check if given file is valid                    
            if os.path.isfile(os.path.normpath(path)+'\\'+img_name) == False:
                raise FileNotFoundError("no such file detected in the given path. please input a valid and existing image file")
            
            # recheck annotation format for required functions (e.g. yolo_txt_resize requires annotation format)
            if func!='image_only_resize' and annotation_format==None:
                raise TypeError("invalid input type in annotation_format! expected input is of type string (i.e. 'polygon-json') got none.")
                
            # check scale
            if type(scale)==tuple and len(scale)==2:
                self.type_of_adjust = 3
            elif type(scale)==int or type(scale)==float:
                if scale>3 or scale<=0:
                    raise ValueError("overlapping scale detected! limit is between 0-3")
                else:
                    self.type_of_adjust = 0
            elif type(scale)==str:
                if scale.lower() == "up" or scale.lower() == "down":
                    if w_limit==None and h_limit==None:
                        raise Exception("scale up or down requires at least one limit value, either width or height limit or could be both") 
                    else:
                        if scale.lower() == "up":
                            self.type_of_adjust = 1
                        else:
                            self.type_of_adjust = 2
                else:
                    raise ValueError("invalid string value in scale! expected input is either 'up' or 'down' only")
            else:
                raise TypeError("invalid scale format! expected input are of type int, float, string ('up' or 'down') and tuple for fix resize (i.e. 0.8 or (200,200))")

            # check width and height limit
            if w_limit!=None or h_limit!=None:
                if type(scale)!=str:
                    raise Exception("invalid pass! width and height limit is only applicable during scale 'up' and 'down'")
                if w_limit!=None:
                    if type(w_limit)!=int:
                        raise TypeError("invalid input type in w_limit. expected input is of type int")
                if h_limit!=None:
                    if type(h_limit)!=int:
                        raise TypeError("invalid input type in h_limit. expected input is of type int")
            
            # check new name
            if new_name==None:
                new_name="resize_"+img_name
            else:
                if type(new_name)!=str:
                    raise TypeError("invalid input type in new_name! expected input is of type string")
                    
            # set check_by to prevent checking repition in child functions
            self.check_by = func      
            
            return path,img_name,scale,annotation_format,new_name,save_path,w_limit,h_limit,save_image_as
        
        elif func=='polygon_or_bbox_json' or func=='yolo_txt' or func=='pascal_voc_xml' or func=='image_only':
            
            # check image name
            if type(img_name)!=str:
                raise TypeError("invalid input type in img_name! expected input is of type string")
                
            # check if given file is valid                    
            if os.path.isfile(os.path.normpath(path)+'\\'+img_name) == False:
                raise FileNotFoundError("no such file detected in the given path. please input a valid and existing image file")
                
            # recheck annotation format for required functions (e.g. yolo_txt_resize requires annotation format)
            if func!='image_only' and annotation_format==None:
                raise TypeError("invalid input type in annotation_format! expected input is of type string (i.e. 'polygon-json') got none.")
            
            # check new name
            if new_name==None:
                new_name="augment_"+img_name
            else:
                if type(new_name)!=str:
                    raise TypeError("invalid input type in new_name! expected input is of type string")
                    
            # check negative points
            if n_points_exist==None:
                n_points_exist==True
            else:
                if type(n_points_exist)!=bool:
                    raise TypeError("invalid input type in n_points_exist! expected input is of type boolean")
                    
            # check transform ang get predefined transformation base on annotation format if none given
            if transform==None:
                transform = self.get_transformation(annotation_format, random_seed)
            elif str(type(transform))!="<class 'albumentations.core.composition.Compose'>":
                raise TypeError("invalid transform given! expected input is of type 'albumentations.core.composition.Compose'")

            # set check_by to prevent checking repition in child functions
            self.check_by = func   
            
            return path,img_name,annotation_format,transform,save_image_as,save_path,new_name,n_points_exist
        elif func == 'get_transformation':
            # check annotation format
            if annotation_format!=None:
                if type(annotation_format)!=str:   
                    raise TypeError("invalid input type in annotation_format! expected input is of type string (i.e. 'polygon-json')")
            # check random seed
            if random_seed!=None:
                if type(random_seed)!=int:
                    raise TypeError("invalid input type in random_seed! expected input is of type int (i.e. 42)") 
                
            return annotation_format,random_seed
            
        elif func == 'set_transformation':
            if transform==None:
                self.transform=None
            else:
                # check transformation
                if str(type(transform))!="<class 'albumentations.core.composition.Compose'>":
                    raise TypeError("invalid  transform given! expected input is of type 'albumentations.core.composition.Compose'")            
                # check random seed
                if random_seed!=None:
                    if type(random_seed)!=int:
                        raise TypeError("invalid input type in random_seed! expected input is of type int (i.e. 42)")
                    
            return transform, random_seed
        
        elif func == 'compute_scale_adjustment':
            # check scale
            if type(scale)==tuple and len(scale)==2:
                self.type_of_adjust = 3
            elif type(scale)==int or type(scale)==float:
                if scale>3 or scale<=0:
                    raise ValueError("overlapping scale detected! limit is between 0-3")
                else:
                    self.type_of_adjust = 0
            elif type(scale)==str:
                if scale.lower() == "up" or scale.lower() == "down":
                    if w_limit==None and h_limit==None:
                        raise Exception("scale up or down requires at least one limit value, either width or height limit or could be both") 
                    else:
                        if scale.lower() == "up":
                            self.type_of_adjust = 1
                        else:
                            self.type_of_adjust = 2
                else:
                    raise ValueError("invalid string value in scale! expected input is either 'up' or 'down' only")
            else:
                raise TypeError("invalid scale format! expected input are of type int, float, string ('up' or 'down') and tuple for fix resize (i.e. 0.8 or (200,200))")

            # check width and height limit
            if w_limit!=None or h_limit!=None:
                if type(scale)!=str:
                    raise Exception("invalid pass! width and height limit is only applicable during scale 'up' and 'down'")
                if w_limit!=None:
                    if type(w_limit)!=int:
                        raise TypeError("invalid input type in w_limit. expected input is of type int")
                if h_limit!=None:
                    if type(h_limit)!=int:
                        raise TypeError("invalid input type in h_limit. expected input is of type int")
            # check width and height
            if type(w)!=int:
                raise TypeError("invalid input type in w (width dimension). expected input is of type int") 
            if type(h)!=int:
                raise TypeError("invalid input type in h (height dimension). expected input is of type int")
            
            # set check_by to prevent checking repition in child functions
            self.check_by = func
            
            return w,h,scale,w_limit,h_limit

    def adjust_img_and_annotate(self,path,scale,w_limit=None,h_limit=None,save_image_as=None,except_list=None,
                                accept_img_format=None,annotation_format=None,save_path=None):
        '''
        List of Parameters
        
        path = directory of images and annotations
        scale = the scaling method of images and annotations. could be int or float (1=original size), 
                tuple(for fix size i.e. (w=300,h=300)) and string('up' and 'down', for auto scaling - requires atleast one limit value)
        h_limit = height limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the height limit as maximum height. 
                  if the original image height is higher than the height limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down'
        w_limit = width limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the width limit as maximum width. 
                  if the original image width is higher than the width limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down' 
        annotation_format = annotation format for annotation adjustments  (e.g. 'polygon_json').
        save path = the saving path of resize images and annotations. if None given, augmentation will be saved on adjust_img_and_annotate
                    folder in the current directory
        save_image_as = the augmented image format. if "default" given, original image format will be used
        except_list = a list of file names (images) that will be excluded in the given path
        accept_img_format = acceptable image formats. type tuple. default values includes '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
        '''
        # check inputs
        path,scale,w_limit,h_limit,save_image_as,except_list,accept_img_format,annotation_format,save_path=self.check_inputs(func="adjust_img_and_annotate",
                                                                                         path=path,scale=scale,w_limit=w_limit,h_limit=h_limit,
                                                                                         save_image_as=save_image_as,except_list=except_list,
                                                                                         accept_img_format=accept_img_format,
                                                                                         annotation_format=annotation_format,
                                                                                         save_path=save_path)
        # for later use
        is_ann_true = False


        # extract all images
        img_names = []
        filenames = os.listdir(path)
        for name in filenames:
            if name.lower().endswith(accept_img_format):
                if str(name) not in except_list:
                    img_names.append(name)      
        total_images = len(img_names)


        if annotation_format==None:
            print("Resizing Images\n")
        else:
            print("Resizing Images and Annotations\n")
            is_ann_true=True

        # start execution time
        stop_threads = False
        percent_queue = queue.Queue()
        countdown_thread = threading.Thread(target = self.exec_time,  args = [lambda : stop_threads, percent_queue])
        countdown_thread.start()

        # for augmentation
        percent_count = 1
        resize_count = 0

        for img_name in img_names:

            # the new name
            new_name = "resize_"+img_name      

            # adjust image and annotations                    
            if annotation_format=="polygon-json" or annotation_format=="bbox-json":
                self.polygon_or_bbox_json_resize(path,img_name,scale,annotation_format,new_name,save_path,w_limit,h_limit,save_image_as)
            elif annotation_format=="yolo-txt":
                self.yolo_txt_resize(path,img_name,scale,annotation_format,new_name,save_path,w_limit,h_limit,save_image_as)
            elif annotation_format=="pascal_voc-xml":
                self.pascal_voc_xml_resize(path,img_name,scale,annotation_format,new_name,save_path,w_limit,h_limit,save_image_as)
            elif annotation_format==None:
                self.image_only_resize(path,img_name,scale,new_name,save_path,w_limit,h_limit,save_image_as)
                
            # update percentage
            percent_queue.put(round(percent_count/total_images * 100,1))
            resize_count+=1
            percent_count+=1

        # stop execution time    
        stop_threads = True
        countdown_thread.join()
        
        # reset check by for input checking
        self.check_by=''
        
        # for count and annotation exist check
        s1,s2 = "",""
        temp = ""                        
        if resize_count>1:
            s1 = "s"
        if is_ann_true:
            temp = " and annotation"
            s2 = "s"
            
        # for printing save path
        if save_path.find(':')==True:  
            print("\n"+str(resize_count)+" resized image{}{}{} was saved at: ".format(s1,temp,s2)+save_path+"\n")
        else:
            print("\n"+str(resize_count)+" resized image{}{}{} was saved at: ".format(s1,temp,s2)+os.getcwd()+"\\"+save_path+"\n")
            
    def image_only_resize(self,path,img_name,scale,new_name=None,save_path=None,w_limit=None,h_limit=None,save_image_as=None):
        '''
        List of Parameters
        
        path = directory of images and annotations
        scale = the scaling method of images and annotations. could be int or float (1=original size), 
                tuple(for fix size i.e. (w=300,h=300)) and string('up' and 'down', for auto scaling - requires atleast one limit value)
        h_limit = height limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the height limit as maximum height. 
                  if the original image height is higher than the height limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down'
        w_limit = width limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the width limit as maximum width. 
                  if the original image width is higher than the width limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down' 
        save path = the saving path of resize images and annotations. if None given, augmentation will be saved on adjust_img_and_annotate
                    folder in the current directory
        save_image_as = the augmented image format. if "default" given, original image format will be used
        img_name = the image name in the given path (full name with extension)
        new_name = the additional prefix name in original name. default name is 'augment_'
        '''
        # check inputs
        parent_functions = ['adjust_img_and_annotate']
        if self.check_by not in parent_functions:
            path,img_name,scale,new_name,save_path,w_limit,h_limit,save_image_as = self.check_inputs(func="image_only_resize",path=path,
                                                                                                                       img_name=img_name,scale=scale,
                                                                                                                       new_name=new_name,save_path=save_path,
                                                                                                                       w_limit=w_limit,h_limit=h_limit,
                                                                                                                       save_image_as=save_image_as)
        # read image
        img = cv2.imread(path+"/"+img_name)
        h, w, _ = img.shape
        # compute image and annotations adjustments
        new_w, new_h, _ = self.compute_scale_adjustment(w=w,h=h,scale=scale,w_limit=w_limit,h_limit=h_limit)
        pure_name = img_name.split('.')[0]

        # resize image
        resize_img = cv2.resize(img, (new_w,new_h))
        # write image                        
        if save_image_as == None:
            cv2.imwrite(save_path+"/"+new_name,resize_img)
        else:
            cv2.imwrite(save_path+"/"+new_name.split('.')[0]+save_image_as[0],resize_img)
            
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return True
            
    def polygon_or_bbox_json_resize(self,path,img_name,scale,annotation_format,new_name=None,save_path=None,w_limit=None,h_limit=None,save_image_as=None):
        '''
        List of Parameters
        
        path = directory of images and annotations
        scale = the scaling method of images and annotations. could be int or float (1=original size), 
                tuple(for fix size i.e. (w=300,h=300)) and string('up' and 'down', for auto scaling - requires atleast one limit value)
        h_limit = height limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the height limit as maximum height. 
                  if the original image height is higher than the height limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down'
        w_limit = width limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the width limit as maximum width. 
                  if the original image width is higher than the width limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down' 
        annotation_format = annotation format for annotation adjustments  (e.g. 'polygon_json').
        save path = the saving path of resize images and annotations. if None given, augmentation will be saved on adjust_img_and_annotate
                    folder in the current directory
        save_image_as = the augmented image format. if "default" given, original image format will be used
        img_name = the image name in the given path (full name with extension)
        new_name = the additional prefix name in original name. default name is 'augment_'
        '''
        # check inputs
        parent_functions = ['adjust_img_and_annotate']
        if self.check_by not in parent_functions:
            path,img_name,scale,annotation_format,new_name,save_path,w_limit,h_limit,save_image_as = self.check_inputs(func="polygon_or_bbox_json_resize",path=path,
                                                                                                                       img_name=img_name,scale=scale,
                                                                                                                       annotation_format=annotation_format,
                                                                                                                       new_name=new_name,save_path=save_path,
                                                                                                                       w_limit=w_limit,h_limit=h_limit,
                                                                                                                       save_image_as=save_image_as)
        # read image
        img = cv2.imread(path+"/"+img_name)
        h, w, _ = img.shape
        # compute image and annotations adjustments
        new_w, new_h, adjust_values = self.compute_scale_adjustment(w=w,h=h,scale=scale,w_limit=w_limit,h_limit=h_limit)
        pure_name = img_name.split('.')[0]

        # resize image
        resize_img = cv2.resize(img, (new_w,new_h))

        # write image                        
        if save_image_as == None:
            cv2.imwrite(save_path+"/"+new_name,resize_img)
        else:
            cv2.imwrite(save_path+"/"+new_name.split('.')[0]+save_image_as[0],resize_img) 
            
        # read annotation
        with open(path+"/"+pure_name+"."+annotation_format.split("-")[1],"r") as f:
            data = json.load(f)

        points = []
        new_points = [] 
        temp = None

        for i in range(0, len(data['shapes'])):
            points.append(data['shapes'][i]['points'])

        # adjust points
        for i in range(0,len(points)):
            if self.type_of_adjust==3:
                temp = [[j[i]*adjust_values[0] if i==0 else j[i]*adjust_values[1] for i in range(0,2)] for j in points[i]]
            else:
                temp = [[i*adjust_values[0] for i in j] for j in points[i]]
            new_points.append(temp)

        # replace old points with the adjusted points
        for i in range(0, len(data['shapes'])):
            data['shapes'][i]['points'] = new_points[i]

        # get all new data of the resize image
        if save_image_as == None:
            with open(save_path+"/"+new_name,'rb') as f:
                imgData = f.read()
        else:
            with open(save_path+"/"+new_name.split('.')[0]+save_image_as[0],'rb') as f:
                imgData = f.read()
        imageData = base64.encodebytes(imgData).decode('utf-8')
        data['imageData'] = imageData
        data['imageHeight'] = new_h
        data['imageWidth'] = new_w
        if save_image_as == None:
            data['imagePath'] = new_name
        else:
            data['imagePath'] = new_name.split('.')[0]+save_image_as[0]

        # save the adjusted annotation
        with open(save_path+"/"+new_name.split(".")[0]+"."+annotation_format.split("-")[1],'w') as f:
            json.dump(data,f, indent=4)
            
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return True
    
    def yolo_txt_resize(self,path,img_name,scale,annotation_format,new_name=None,save_path=None,w_limit=None,h_limit=None,save_image_as=None):    
        '''
        List of Parameters
        
        path = directory of images and annotations
        scale = the scaling method of images and annotations. could be int or float (1=original size), 
                tuple(for fix size i.e. (w=300,h=300)) and string('up' and 'down', for auto scaling - requires atleast one limit value)
        h_limit = height limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the height limit as maximum height. 
                  if the original image height is higher than the height limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down'
        w_limit = width limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the width limit as maximum width. 
                  if the original image width is higher than the width limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down' 
        annotation_format = annotation format for annotation adjustments  (e.g. 'polygon_json').
        save path = the saving path of resize images and annotations. if None given, augmentation will be saved on adjust_img_and_annotate
                    folder in the current directory
        save_image_as = the augmented image format. if "default" given, original image format will be used
        img_name = the image name in the given path (full name with extension)
        new_name = the additional prefix name in original name. default name is 'augment_'
        '''
        # check inputs
        parent_functions = ['adjust_img_and_annotate']
        if self.check_by not in parent_functions:
            path,img_name,scale,annotation_format,new_name,save_path,w_limit,h_limit,save_image_as = self.check_inputs(func="yolo_txt_resize",path=path,
                                                                                                                       img_name=img_name,scale=scale,
                                                                                                                       annotation_format=annotation_format,
                                                                                                                       new_name=new_name,save_path=save_path,
                                                                                                                       w_limit=w_limit,h_limit=h_limit,
                                                                                                                       save_image_as=save_image_as)
        # read image
        img = cv2.imread(path+"/"+img_name)
        h, w, _ = img.shape

        # compute image and annotations adjustments   
        new_w, new_h, _ = self.compute_scale_adjustment(w=w,h=h,scale=scale,w_limit=w_limit,h_limit=h_limit)
        pure_name = img_name.split('.')[0]

        # resize image
        resize_img = cv2.resize(img, (new_w,new_h))

        # write image                        
        if save_image_as == None:
            cv2.imwrite(save_path+"/"+new_name,resize_img)
        else:
            cv2.imwrite(save_path+"/"+new_name.split('.')[0]+save_image_as[0],resize_img)
            
        # copy annotations class
        if os.path.isfile(save_path+"/"+"classes.txt")==False:
            copyfile(path+"classes.txt",save_path+"/"+"classes.txt")

        # read annotation
        with open(path+pure_name+"."+annotation_format.split("-")[1],'r') as file:
            annotation = file.read()

        data = annotation.split('\n')
        dw, dh = new_w, new_h      
        category_ids = []
        new_annotation = ""

        # adjust coordinates
        for i in range(0,len(data)-1):
            length = len(data[i].split(' '))
            annotation = data[i].split(' ')[length-4:length]
            default_labels = " ".join(data[i].split(' ')[0:length-4])
            x, y, w, h = float(annotation[0]), float(annotation[1]), float(annotation[2]), float(annotation[3])
            x2 = round((x - w / 2) * dw)
            x1 = round((x + w / 2) * dw)
            y2 = round((y - h / 2) * dh)
            y1 = round((y + h / 2) * dh)
            wdw, hdh = round(w*dw), round(h*dh)
            new_ann = " ".join([str(round(i,6)) for i in [((x2 + x1) / 2) / dw, ((y2 + y1) / 2) / dh, wdw / dw, hdh / dh]])        
            new_annotation+=default_labels+" "+new_ann+"\n"

        # save the adjusted annotation
        with open(save_path+"/"+new_name.split(".")[0]+"."+annotation_format.split("-")[1],'w') as file:
            file.write(new_annotation)
        
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return True
    
    def pascal_voc_xml_resize(self,path,img_name,scale,annotation_format,new_name=None,save_path=None,w_limit=None,h_limit=None,save_image_as=None): 
        '''
        List of Parameters
        
        path = directory of images and annotations
        scale = the scaling method of images and annotations. could be int or float (1=original size), 
                tuple(for fix size i.e. (w=300,h=300)) and string('up' and 'down', for auto scaling - requires atleast one limit value)
        h_limit = height limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the height limit as maximum height. 
                  if the original image height is higher than the height limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down'
        w_limit = width limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the width limit as maximum width. 
                  if the original image width is higher than the width limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down' 
        annotation_format = annotation format for annotation adjustments  (e.g. 'polygon_json').
        save path = the saving path of resize images and annotations. if None given, augmentation will be saved on adjust_img_and_annotate
                    folder in the current directory
        save_image_as = the augmented image format. if "default" given, original image format will be used
        img_name = the image name in the given path (full name with extension)
        new_name = the additional prefix name in original name. default name is 'augment_'
        '''
        # check inputs
        parent_functions = ['adjust_img_and_annotate']
        if self.check_by != 'adjust_img_and_annotate':
            path,img_name,scale,annotation_format,new_name,save_path,w_limit,h_limit,save_image_as = self.check_inputs(func="pascal_voc_xml_resize",path=path,
                                                                                                                       img_name=img_name,scale=scale,
                                                                                                                       annotation_format=annotation_format,
                                                                                                                       new_name=new_name,save_path=save_path,
                                                                                                                       w_limit=w_limit,h_limit=h_limit,
                                                                                                                       save_image_as=save_image_as)
        # read image
        img = cv2.imread(path+"/"+img_name)
        h, w, _ = img.shape

        # compute image and annotations adjustments   
        new_w, new_h, adjust_values = self.compute_scale_adjustment(w=w,h=h,scale=scale,w_limit=w_limit,h_limit=h_limit)
        pure_name = img_name.split('.')[0]  

        # resize image
        resize_img = cv2.resize(img, (new_w,new_h))

        # write image                        
        if save_image_as == None:
            cv2.imwrite(save_path+"/"+new_name,resize_img)
        else:
            cv2.imwrite(save_path+"/"+new_name.split('.')[0]+save_image_as[0],resize_img)
        
        # read annotation
        mytree = ET.parse(path+pure_name+"."+annotation_format.split("-")[1])
        bboxes = []
        newbboxes = []
        category_ids = []

        for obj in mytree.iter('object'):
            # extract coordinates
            xmin = int(obj.find(".//xmin").text)
            ymin = int(obj.find(".//ymin").text)
            xmax = int(obj.find(".//xmax").text)
            ymax = int(obj.find(".//ymax").text)
            category_ids.append(obj.find(".//name").text)
            bboxes.append([xmin,ymin,xmax,ymax])

        # adjust coordinates
        if self.type_of_adjust==3:
            for bbox in bboxes:
                xmin, xmax = bbox[0]*adjust_values[0], bbox[2]*adjust_values[0]
                ymin, ymax = bbox[1]*adjust_values[1], bbox[3]*adjust_values[1]
                newbboxes.append([xmin,ymin,xmax,ymax])
        else:
            for bbox in bboxes:
                temp = [x*adjust_values[0] for x in bbox]
                newbboxes.append(temp)

        count = 0
        # set new values
        for obj in mytree.iter('object'):
            # set new values
            obj.find(".//xmin").text = str(round(newbboxes[count][0]))
            obj.find(".//ymin").text = str(round(newbboxes[count][1]))
            obj.find(".//xmax").text = str(round(newbboxes[count][2]))
            obj.find(".//ymax").text = str(round(newbboxes[count][3]))
            obj.find(".//name").text = category_ids[count]
            count+=1
        mytree.find(".//width").text = str(new_w)
        mytree.find(".//height").text = str(new_h)
        mytree.find(".//folder").text = os.path.basename(save_path)
        if save_image_as == None:
            mytree.find(".//filename").text = new_name
        else:
            mytree.find(".//filename").text = new_name.split('.')[0]+save_image_as[0]
        if save_path.find(':')==True:
            if save_image_as == None:
                mytree.find(".//path").text = save_path+"\\"+new_name
            else:
                mytree.find(".//path").text = save_path+"\\"+new_name.split('.')[0]+save_image_as[0]
        else:
            if save_image_as == None:
                mytree.find(".//path").text = os.getcwd()+"\\"+save_path+"\\"+new_name
            else:
                mytree.find(".//path").text = os.getcwd()+"\\"+save_path+"\\"+new_name.split('.')[0]+save_image_as[0]

        # write new adjusted annotations
        mytree.write(save_path+"/"+new_name.split('.')[0]+"."+annotation_format.split("-")[1])
        
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return True
    
    def compute_scale_adjustment(self,w,h,scale,w_limit=None,h_limit=None):
        '''
        List of Parameters
        
        h = height
        w = width
        scale = the scaling method of images and annotations. could be int or float (1=original size), 
                tuple(for fix size i.e. (w=300,h=300)) and string('up' and 'down', for auto scaling - requires atleast one limit value)
        h_limit = height limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the height limit as maximum height. 
                  if the original image height is higher than the height limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down'
        w_limit = width limit. Applicable only during scale 'up' or 'down' value scale. 
                  if scale is assigned to 'up', all images will be scaled up with the width limit as maximum width. 
                  if the original image width is higher than the width limit. The image will not be scaled
                  and will retain its original dimension. The opposite goes during scaling 'down' 
        '''
        # check inputs
        parent_functions = ['polygon_or_bbox_json_resize','yolo_txt_resize','pascal_voc_xml_resize','adjust_img_and_annotate','image_only_resize']
        if self.check_by not in parent_functions:
            w,h,scale,w_limit,h_limit = self.check_inputs(func="compute_scale_adjustment",w=w,h=h,scale=scale,w_limit=w_limit,h_limit=h_limit)
            
        new_w, new_h = 0, 0
        adjust_values = []

        # for up scale
        if self.type_of_adjust==1:
            if w_limit!=None and h_limit==None:
                if w_limit>=w:
                    scale = w_limit/w
                else:
                    scale = 1
            elif w_limit==None and h_limit!=None:
                if h_limit>=h:
                    scale = h_limit/h
                else:
                    scale = 1
            elif w_limit!=None and h_limit!=None:
                if w_limit>=w or h_limit>=h:
                    scale = max(w_limit/w, h_limit/h)
                else:
                    scale = 1 
        # for down scale
        elif  self.type_of_adjust==2:
            if w_limit!=None and h_limit==None:
                if w_limit<=w:
                    scale = w_limit/w
                else:
                    scale = 1 
            elif w_limit==None and h_limit!=None:
                if h_limit<=h:
                    scale = h_limit/h
                else:
                    scale = 1 
            elif w_limit!=None and h_limit!=None:
                if w_limit<=w or h_limit<=h:
                    scale = min(w_limit/w, h_limit/h)
                else:
                    scale = 1

        # for fix scale adjustment
        if  self.type_of_adjust==3:
            new_w = scale[0]
            new_h = scale[1]

            # get annotation adjust values
            adjust_values.append(scale[0]/w)
            adjust_values.append(scale[1]/h)

        # for fix scale using int, float, up, and down adjustments 
        else:
            new_w = round(w*scale)
            new_h = round(h*scale)

            # get annotation adjust values
            adjust_values.append(scale)
            
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return new_w, new_h, adjust_values
        
    def A_augmentation(self,path,transform=None,annotation_format=None,save_image_as=None,except_list=None, 
                       accept_img_format=None,random_seed=None,n_points_exist=None,save_path=None):
        '''
        List of Parameters
        
        path = directory of images and annotations
        annotation_format = annotation format for augmentation (e.g. 'polygon_json'). if string none given
                            only images will be augmented
        save_image_as = the augmented image format. if "default" given, original image format will be used
        random_seed = random seed. default value None
        n_points_exist = if negatives keypoints exist in annotation. if true given negative points is change to 0,
                         recommended to remain to true as albumentation does not support negative points
        save path = the saving path of augmented images and annotations. if None given, augmentation will be saved on albumentation_augmentation
                    folder in the current directory
        accept_img_format = acceptable image formats. type tuple. default values includes '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
        except_list = a list of file names (images) that will be excluded in the given path
        ''' 
        path,transform,annotation_format,save_image_as,except_list,accept_img_format,random_seed,n_points_exist,save_path=self.check_inputs(func="A_augmentation",
                                                                                                                                       path=path,
                                                                                                                                       transform=transform,
                                                                                                                                       annotation_format=annotation_format,
                                                                                                                                       save_image_as=save_image_as,
                                                                                                                                       except_list=except_list,
                                                                                                                                       accept_img_format=accept_img_format,
                                                                                                                                       random_seed=random_seed,
                                                                                                                                       n_points_exist=n_points_exist,
                                                                                                                                       save_path=save_path)
        # for later use                                                                                                                               
        is_ann_true = False
            
        # extract all images
        img_names = []
        filenames = os.listdir(path)      
        for name in filenames:
            if name.lower().endswith(accept_img_format):
                if str(name) not in except_list:
                    img_names.append(name)       
        total_images = len(img_names)
        
        if annotation_format==None:
            print("Augmenting Images\n")
        else:
            print("Augmenting Images and Annotations \n")
            is_ann_true = True
            
        # start execution time
        stop_threads = False
        percent_queue = queue.Queue()
        countdown_thread = threading.Thread(target = self.exec_time,  args = [lambda : stop_threads, percent_queue])
        countdown_thread.start()

        # for augmentation
        percent_count = 1
        augment_count = 0


        for img_name in img_names:

            # the new name
            new_name = "augment_"+img_name

            # augment images and annotations
            if annotation_format == None:
                self.image_only(path,img_name,transform,save_image_as,save_path,new_name)
            elif annotation_format == "polygon-json" or annotation_format == "bbox-json":
                self.polygon_or_bbox_json(path,img_name,annotation_format,transform,save_image_as,save_path,new_name,n_points_exist)
            elif annotation_format == "yolo-txt":
                self.yolo_txt(path,img_name,annotation_format,transform,save_image_as,save_path,new_name)
            elif annotation_format == "pascal_voc-xml":
                self.pascal_voc_xml(path,img_name,annotation_format,transform,save_image_as,save_path,new_name)

            # update percentage
            percent_queue.put(round(percent_count/total_images * 100,1))
            augment_count+=1
            percent_count+=1

        # stop execution time    
        stop_threads = True
        countdown_thread.join()
        
        # reset check by for input checking
        self.check_by=''
        
        # for count and annotation exist check
        s1,s2 = "",""
        temp = ""                        
        if augment_count>1:
            s1 = "s"
        if is_ann_true:
            temp = " and annotation"
            s2 = "s"
        
        # for printing save path
        if save_path.find(':')==True:  
            print("\n"+str(augment_count)+" augmented image{}{}{} was saved at: ".format(s1,temp,s2)+save_path+"\n")
        else:
            print("\n"+str(augment_count)+" augmented image{}{}{} was saved at: ".format(s1,temp,s2)+os.getcwd()+"\\"+save_path+"\n")

    
    def image_only(self,path,img_name,transform=None,save_image_as=None,save_path=None,new_name=None):
        '''
        List of Parameters
        
        path = directory of images and annotations
        save_image_as = the augmented image format. if "default" given, original image format will be used
        random_seed = random seed. default value None
        n_points_exist = if negatives keypoints exist in annotation. if true given negative points is change to 0,
                         recommended to remain to true as albumentation does not support negative points
        save path = the saving path of augmented images and annotations. if None given, augmentation will be saved on albumentation_augmentation
                    folder in the current directory
        img_name = the image name in the given path (full name with extension)
        new_name = the additional prefix name in original name. default name is 'augment_'
        '''
        parent_functions = ['A_augmentation']
        if self.check_by not in parent_functions: 
            path,img_name,_,transform,save_image_as,save_path,new_name,_=self.check_inputs(func="image_only",path=path,
                                                                                        img_name=img_name,
                                                                                        transform=transform,
                                                                                        save_image_as=save_image_as,
                                                                                        save_path=save_path,new_name=new_name)
        # read image
        img = cv2.imread(path+"/"+img_name)       
        # augment image
        aug_image = transform(image=img)['image']

        # write the augmented image
        if save_image_as == None:
            cv2.imwrite(save_path+"/"+new_name,aug_image)
        else:
            cv2.imwrite(save_path+"/"+new_name.split('.')[0]+save_image_as[0],aug_image)
            
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return True
    
    def polygon_or_bbox_json(self,path,img_name,annotation_format,transform=None,save_image_as=None,save_path=None,new_name=None,n_points_exist=None):
        '''
        List of Parameters
        
        path = directory of images and annotations
        annotation_format = annotation format for augmentation  (e.g. 'polygon_json'). if string none given
                            only images will be augmented
        save_image_as = the augmented image format. if "default" given, original image format will be used
        random_seed = random seed. default value None
        n_points_exist = if negatives keypoints exist in annotation. if true given negative points is change to 0,
                         recommended to remain to true as albumentation does not support negative points
        save path = the saving path of augmented images and annotations. if None given, augmentation will be saved on albumentation_augmentation
                    folder in the current directory
        img_name = the image name in the given path (full name with extension)
        new_name = the additional prefix name in original name. default name is 'augment_'
        '''
        parent_functions = ['A_augmentation']
        if self.check_by not in parent_functions: 
            path,img_name,annotation_format,transform,save_image_as,save_path,new_name,n_points_exist=self.check_inputs(func="polygon_or_bbox_json",path=path,
                                                                                                                    img_name=img_name,
                                                                                                                    annotation_format=annotation_format,
                                                                                                                    transform=transform,
                                                                                                                    save_image_as=save_image_as,
                                                                                                                    save_path=save_path,new_name=new_name,
                                                                                                                    n_points_exist=n_points_exist)
        # read image
        img = cv2.imread(path+"/"+img_name)
        # get annotation ext
        annotation_ext = annotation_format.split('-')[1]
        # pure name
        pure_name = img_name.split(".")[0]
        
        # extract keypoints from annotation
        with open(path+"/"+pure_name+"."+annotation_ext,'r') as file:
            data = json.load(file)
        keyPoints = []
        lengths = []
        num_of_annotations = len(data['shapes'])
        for i in range(0,num_of_annotations):
            keyPoints.extend(data['shapes'][i]['points'])
            lengths.append(len(data['shapes'][i]['points']))

        # convert negative points to 0
        if n_points_exist == True:
            keyPoints = [[x*0 if x<0 else x for x in y] for y in keyPoints]

        # transform keypoints to list of tuple (recommended format for keypoints augmentation)
        keypoints = list(tuple(x) for x in keyPoints)
        
        # augment images
        transformed = transform(image=img, keypoints=keypoints)
        
        # write the augmented image
        if save_image_as == None:
            cv2.imwrite(save_path+"/"+new_name,transformed['image'])
        else:
            cv2.imwrite(save_path+"/"+new_name.split('.')[0]+save_image_as[0],transformed['image'])

        # transform keypoints back into list of list (original json format)
        list_of_lists = list(list(x) for x in transformed['keypoints'])

        # get image data of augmented image
        if save_image_as == None:
            with open(save_path+'/'+new_name, mode='rb') as file:
                img = file.read()
            imageData = base64.encodebytes(img).decode('utf-8')
        else:
            with open(save_path+'/'+new_name.split('.')[0]+save_image_as[0], mode='rb') as file:
                img = file.read()
            imageData = base64.encodebytes(img).decode('utf-8')
            
        # get height and width of augmented image
        h, w, _ = transformed['image'].shape
        
        # keep track of invisible points (points that will be excluded)
        tracker = [(i[0] < 0 or i[1] < 0 or i[0] >= w or i[1] >= h) for i in transformed['keypoints']]
        
        # separate keypoints (in case of 2 or more annotations)
        annotation_lists = []
        count, limit = 0, 0 
        temp = []
        for i in list_of_lists:
            if tracker[count]==False:
                temp.append(i)
            count+=1
            if count == lengths[limit]:
                annotation_lists.append(temp)
                count=0
                limit+=1
                temp = []

        # assign new values
        for i in range(0, len(annotation_lists)):
            data['shapes'][i]['points'] = annotation_lists[i]
        if save_image_as == None:
            data['imagePath'] = new_name
        else:
            data['imagePath'] = new_name.split('.')[0]+save_image_as[0]
        data['imageData'] = imageData
        data['imageHeight'] = h
        data['imageWidth'] = w

        # write the json file
        with open(save_path+"/"+new_name.split('.')[0]+"."+annotation_ext,"w") as f:
            json.dump(data,f, cls=self.NpEncoder, indent=4)
            
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return True
    
    def yolo_txt(self,path,img_name,annotation_format,transform=None,save_image_as=None,save_path=None,new_name=None):
        '''
        List of Parameters
        
        path = directory of images and annotations
        annotation_format = annotation format for augmentation  (e.g. 'polygon_json'). if string none given
                            only images will be augmented
        save_image_as = the augmented image format. if "default" given, original image format will be used
        random_seed = random seed. default value None
        n_points_exist = if negatives keypoints exist in annotation. if true given negative points is change to 0,
                         recommended to remain to true as albumentation does not support negative points
        save path = the saving path of augmented images and annotations. if None given, augmentation will be saved on albumentation_augmentation
                    folder in the current directory
        img_name = the image name in the given path (full name with extension)
        new_name = the additional prefix name in original name. default name is 'augment_'
        '''
        parent_functions = ['A_augmentation']
        if self.check_by not in parent_functions: 
            path,img_name,annotation_format,transform,save_image_as,save_path,new_name,n_points_exist=self.check_inputs(func="yolo_txt",path=path,
                                                                                                                    img_name=img_name,
                                                                                                                    annotation_format=annotation_format,
                                                                                                                    transform=transform,
                                                                                                                    save_image_as=save_image_as,
                                                                                                                    save_path=save_path,new_name=new_name)
        # read image
        img = cv2.imread(path+"/"+img_name)
        # get annotation ext
        annotation_ext = annotation_format.split('-')[1]
        # pure name
        pure_name = img_name.split(".")[0]
        
        # copy annotations class
        if os.path.isfile(save_path+"/"+"classes.txt")==False:
            copyfile(path+"classes.txt",save_path+"/"+"classes.txt")

        # read and extract annotations
        with open(path+pure_name+"."+annotation_ext,'r') as file:
            annotation = file.read()     
        data = annotation.split('\n')

        category_ids = []
        bboxes = []

        for i in range(0,len(data)-1):
            length = len(data[i].split(' '))
            annotation = data[i].split(' ')[length-4:length]
            default_label = " ".join(data[i].split(' ')[0:length-4])
            temp = [float(i) for i in annotation]
            bboxes.append(temp)
            category_ids.append(int(default_label))

        # augment image and annotations
        transformed = transform(image=img,bboxes=bboxes, category_ids=category_ids)

        # write the augmented image
        if save_image_as == None:
            cv2.imwrite(save_path+"/"+new_name,transformed['image'])
        else:
            cv2.imwrite(save_path+"/"+new_name.split('.')[0]+save_image_as[0],transformed['image'])


        # arrange new annnotations to right format
        new_annotation = ""
        for i in range(0,len(transformed['bboxes'])):
            temp = str(category_ids[i])
            for j in range(0, len(transformed['bboxes'][i])):
                temp+=" "+str(round(transformed['bboxes'][i][j],6))
            new_annotation+=temp+"\n"

        # write new annotations
        with open(save_path+"/"+new_name.split('.')[0]+"."+annotation_ext,"w") as f:
            f.write(new_annotation)
            
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return True
    
    def pascal_voc_xml(self,path,img_name,annotation_format,transform=None,save_image_as=None,save_path=None,new_name=None):
        '''
        List of Parameters
        
        path = directory of images and annotations
        annotation_format = annotation format for augmentation  (e.g. 'polygon_json'). if string none given
                            only images will be augmented
        save_image_as = the augmented image format. if "default" given, original image format will be used
        random_seed = random seed. default value None
        n_points_exist = if negatives keypoints exist in annotation. if true given negative points is change to 0,
                         recommended to remain to true as albumentation does not support negative points
        save path = the saving path of augmented images and annotations. if None given, augmentation will be saved on albumentation_augmentation
                    folder in the current directory
        img_name = the image name in the given path (full name with extension)
        new_name = the additional prefix name in original name. default name is 'augment_'
        '''
        parent_functions = ['A_augmentation']
        if self.check_by not in parent_functions: 
            path,img_name,annotation_format,transform,save_image_as,save_path,new_name,n_points_exist=self.check_inputs(func="pascal_voc_xml",path=path,
                                                                                                                    img_name=img_name,
                                                                                                                    annotation_format=annotation_format,
                                                                                                                    transform=transform,
                                                                                                                    save_image_as=save_image_as,
                                                                                                                    save_path=save_path,new_name=new_name)        
        # read image
        img = cv2.imread(path+"/"+img_name)
        # get annotation ext
        annotation_ext = annotation_format.split('-')[1]
        # pure name
        pure_name = img_name.split(".")[0]
        
        # read annotation
        mytree = ET.parse(path+"/"+pure_name+"."+annotation_ext)
        bboxes = []
        category_ids = []

        # extract coordinates
        for obj in mytree.iter('object'):
            category_ids.append(obj.find(".//name").text) 
            xmin = int(obj.find(".//xmin").text)
            ymin = int(obj.find(".//ymin").text)
            xmax = int(obj.find(".//xmax").text)
            ymax = int(obj.find(".//ymax").text)
            bboxes.append([xmin,ymin,xmax,ymax])

        # augment
        transformed = transform(image=img,bboxes=bboxes,category_ids=category_ids)

        # write the augmented image
        if save_image_as == None:
            cv2.imwrite(save_path+"/"+new_name,transformed['image'])
        else:
            cv2.imwrite(save_path+"/"+new_name.split('.')[0]+save_image_as[0],transformed['image'])

        # set new values
        count = 0
        h, w , _ = transformed['image'].shape
        for obj in mytree.iter('object'):
            obj.find(".//xmin").text = str(round(transformed['bboxes'][count][0]))
            obj.find(".//ymin").text = str(round(transformed['bboxes'][count][1]))
            obj.find(".//xmax").text = str(round(transformed['bboxes'][count][2]))
            obj.find(".//ymax").text = str(round(transformed['bboxes'][count][3]))
            obj.find(".//name").text = str(transformed['category_ids'][count])
            count+=1

        mytree.find(".//width").text = str(w)
        mytree.find(".//height").text = str(h)
        mytree.find(".//folder").text = os.path.basename(save_path)
        
        if save_image_as == None:
            mytree.find(".//filename").text = new_name
        else:
            mytree.find(".//filename").text = new_name.split('.')[0]+save_image_as[0]
        if save_path.find(':')==True:
            if save_image_as == None:
                mytree.find(".//path").text = save_path+"\\"+new_name
            else:
                mytree.find(".//path").text = save_path+"\\"+new_name.split('.')[0]+save_image_as[0]
        else:
            if save_image_as == None:
                mytree.find(".//path").text = os.getcwd()+"\\"+save_path+"\\"+new_name
            else:
                mytree.find(".//path").text = os.getcwd()+"\\"+save_path+"\\"+new_name.split('.')[0]+save_image_as[0]
    
        # write new annotations
        mytree.write(save_path+"/"+new_name.split('.')[0]+"."+annotation_ext)
        
        # reset check by for input checking    
        if self.check_by not in parent_functions:
            self.check_by=''
        return True
    
    def set_transformation(self,transform,random_seed=None):
        '''
        List of Parameters
        
        transform = augmentation pipeline
        random_seed = random seed
        '''
        transform, random_seed = self.check_inputs(func="set_transformation",transform=transform,random_seed=random_seed)
        if random_seed!=None:
            random.seed(random_seed)
        self.transform = transform
        # reset check by for input checking    
        self.check_by=''
        return True
    
    def get_transformation(self,annotation_format=None,random_seed=None):
        '''
        List of Parameters
        
        annotation_format = the annotation format (e.g. polygon_json, yolo_txt)
        random_seed = random seed
        '''
        parent_functions = ['A_augmentation','pascal_voc_xml','yolo_txt','polygon_or_bbox_json']
        if self.check_by not in parent_functions: 
            annotation_format, random_seed = self.check_inputs(func="get_transformation",annotation_format=annotation_format,random_seed=random_seed)
        if self.transform == None:
            predefined_transform = None
            if annotation_format == None:
                predefined_transform = A.Compose([
                A.ShiftScaleRotate(p=0.2),
                A.RandomRotate90(),
                A.RandomScale(),
                A.Transpose(),
                A.Flip(),
                A.OneOf([
                    A.GaussNoise(),
                    A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1)
                ], p=0.3),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.3),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    A.PiecewiseAffine(p=0.3)
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),            
                ], p=0.3),
                ])
            elif annotation_format == "polygon-json" or annotation_format == "bbox-json":
                predefined_transform = A.Compose([
                    A.RandomRotate90(),
                    A.RandomScale(),
                    A.Transpose(),
                    A.Flip(),
                    A.OneOf([
                        A.GaussNoise(),
                        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1)
                    ], p=0.4),
                    A.OneOf([
                        A.MotionBlur(p=.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.4),
                    A.OneOf([
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                        A.RandomBrightnessContrast(),            
                    ], p=0.4),
                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            elif annotation_format == "yolo-txt":
                predefined_transform = A.Compose([
                    A.ShiftScaleRotate(),
                    A.RandomRotate90(),
                    A.RandomScale(),
                    A.Transpose(),
                    A.Flip(),
                    A.OneOf([
                        A.GaussNoise(),
                        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1)
                    ], p=0.4),
                    A.OneOf([
                        A.MotionBlur(p=.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.4),
                    A.OneOf([
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                        A.RandomBrightnessContrast(),            
                    ], p=0.4),
                ], bbox_params=A.BboxParams(format="yolo", label_fields=['category_ids']))
            elif annotation_format == "pascal_voc-xml":
                predefined_transform = A.Compose([
                    A.ShiftScaleRotate(),
                    A.RandomRotate90(),
                    A.RandomScale(),
                    A.Transpose(),
                    A.Flip(),
                    A.OneOf([
                        A.GaussNoise(),
                        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1)
                    ], p=0.4),
                    A.OneOf([
                        A.MotionBlur(p=.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.4),
                    A.OneOf([
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                        A.RandomBrightnessContrast(),            
                    ], p=0.4),
                ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['category_ids']))
            if random_seed!=None:
                random.seed(random_seed)
            # reset check by for input checking    
            if self.check_by not in parent_functions:
                self.check_by=''
            return predefined_transform
        else:
            # reset check by for input checking    
            if self.check_by not in parent_functions:
                self.check_by=''
            return self.transform