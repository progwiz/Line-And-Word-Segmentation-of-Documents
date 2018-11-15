import global_vars
import warnings
import os
import numpy as np
import cv2
import segmenter_final
from time import time
import matplotlib as mpl
import words
import preprocessor
from matplotlib import pyplot

warnings.filterwarnings("ignore")     
t0=time() 
main_fig=mpl.pyplot.figure(0)
current_dir=os.path.dirname(__file__)

#change input directory to directory containing document image file
input_directory="G:\Academic and Work\Database\Test"
    
for fileno,fn in enumerate(os.listdir(input_directory)):

    #change fn to the document filename
    if fn!="sample2.jpg": continue
    filename=os.path.join(input_directory,fn)
    
    # read image and resize
    t_initial=time()
    init_img=cv2.imread(filename,0)    
    higher_dimension=max([init_img.shape[0],init_img.shape[1]])
    sf=higher_dimension/1000
    if sf>1:
        init_img=cv2.resize(init_img,None,fx=1/sf,fy=1/sf,interpolation=cv2.INTER_AREA)

    # binarize image
    bin_img=preprocessor.binarize(init_img)
    print ("reading and binarization done")
    print ("reading and binarization time: ", time()-t_initial)
    
    # noise removal
    t_clean=time()
    bin_img=preprocessor.clean_img(bin_img)
    print (time()-t_clean)
    print ("cleaning complete")
    print ("Cleaning time:", time()-t_clean)
    
    # skew correction
    t_rotate=time()
    rot_set=preprocessor.rotate2(bin_img)
    print ("rotation_angle:",rot_set[0])
    bin_img=rot_set[1]
    print ("rotation complete")
    print ("Rotation time:", time()-t_rotate)
    
    # line segmentation
    t_line=time()
    lines=segmenter_final.segment_lines(bin_img) 
    print("Line segmentation complete")
    print("Line segmentation time", time()-t_line)
    
    # display lines
    temp_img=np.zeros(bin_img.shape,dtype=np.int32)
    temp_img=bin_img.copy()
    for line_no,line in enumerate(lines):
        for comp_no in line.lorw+line.puncts:
            for point in global_vars.comp_dict[comp_no].points:
                temp_img[point[0],point[1]]=line_no+1

    mpl.pyplot.figure(global_vars.figno)
    mpl.pyplot.imshow(temp_img)
    global_vars.figno+=1
    
    
    t_word=time()
    wrds=words.segment_words(lines)
    """
    # display words
    temp_img=np.zeros(bin_img.shape,dtype=np.int32)
    temp_img=bin_img.copy()
    for word_no,word in enumerate(wrds):
            for point in word:
                temp_img[point[0],point[1]]=word_no+1

    mpl.pyplot.figure(global_vars.figno)
    mpl.pyplot.imshow(temp_img)
    global_vars.figno+=1
    """
    print ("Word segmentation complete")
    print("Word segmentation time", time()-t_word)

mpl.pyplot.show()