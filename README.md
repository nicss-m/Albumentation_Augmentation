<h3>Module Name : albumentation_augmentation </h3>
<h3>Description: This module augment images and automatically adjust its annotations using <a href=https://github.com/albumentations-team/albumentations>albumentations</a> package.</h3>

<h3>Dependencies:
	albumentations == 1.0.0
	numpy == 1.19.3
	opencv == 4.4.0.44</h3>

# Brief Documentation:

# List of Augmentation Parameters ###

## Batches and Single
path = directory of images and annotations
annotation_format = annotation format for augmentation  (e.g. 'polygon_json'). if string none given
                    only images will be augmented
save_image_as = the augmented image format. if "default" given, original image format will be used
random_seed = random seed. default value None
n_points_exist = if negatives keypoints exist in annotation. if true given negative points is change to 0,
                 recommended to remain to true as albumentation does not support negative points
save path = the saving path of augmented images and annotations. if None given, augmentation will be saved on albumentation_augmentation
            folder in the current directory

### Batches Only
accept_img_format = acceptable image formats. type tuple. default values includes '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
except_list = a list of file names (images) that will be excluded in the given path

### Single Only
img_name = the image name in the given path (full name with extension)
new_name = the additional prefix name in original name. default name is 'augment_'

# List of Resize Parameters ###

### Batches and Single
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

### Batches Only
except_list = a list of file names (images) that will be excluded in the given path
accept_img_format = acceptable image formats. type tuple. default values includes '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'

### Single Only
img_name = the image name in the given path (full name with extension)
new_name = the additional prefix name in original name. default name is 'augment_'
