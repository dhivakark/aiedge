import os
import cv2
import numpy as np
import json
import shutil
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class dataset ():
  def __init__(self, dsDir, annotDir, categories):
    self.dsDir            = dsDir                                     #String - Path to dataset
    self.annotDir         = annotDir                                  #String - Path to annotations
    self.categories       = categories                                #Tuple of category names in annotation/ dataset
    self.imgFiles         = os.listdir(self.dsDir)                    #list of files in dataset
    self.annFiles         = os.listdir(self.annotDir)                 #list of files in annotations
    self.imgFNames        = [i.split('.')[0] for i in self.imgFiles]  #list of file names w/o extension in dataset - both for img & annotation
    self.dsSize           = len(self.imgFNames)                       #num of images in dataset
    self.itemPerCategory  = dict((x,0) for x in self.categories)      #num of items in corresponding category grouped as dict
    self.lblPerImge       = []                                        #num of labels in each image    
    self.dsImgRes         = []                                        #Resolution of the dataset image
    self.imgHRes          = 1936                                      #Horizontal resolution of training/test images
    self.imgVRes          = 1216                                      #Vertical resolution of training/test images
    self.imgChnl          = 3                                         #number of channels in training/test images

  def rdItemPerCategory (self):
    '''
    This function reads the dataset annotations and identifies how many items
    per class are available in the data set, to show the distribution.
    '''
    for annFile in self.annFiles:
      with open(os.path.join(self.annotDir, annFile)) as f:
        annotation  = json.load(f)      
      labels        = annotation['labels']
      self.lblPerImge.append(len(labels))
      for label in labels:
        self.itemPerCategory[label['category']]+=1
    
  def visualizeDsDistribution (self):
    '''
    Show bar graph of that shows the distribution of each class of object in dataset.
    '''
    self.rdItemPerCategory()
    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    plt.xlabel('Categories')
    plt.ylabel('The Number of Bounding Boxes')
    plt.title('Per Category Distribution')
    bar = plt.bar(range(len(self.itemPerCategory)), list(self.itemPerCategory.values()))
    plt.xticks (range(len(self.itemPerCategory)),list(self.itemPerCategory.keys()))
    plt.show()
    
  def printItemPerCategory(self):
    '''
    Simple print of the distribution of each class of object in dataset.
    '''
    self.rdItemPerCategory()
    for k,v in self.itemPerCategory.items():
      print ('Category: '+k+' --> Number of items: '+str(v))
    
  def printDataSetStats(self):
    '''
    Simple print of the statistics of the dataset.
    '''
    print ('Total number of images in dataset = '+str(self.dsSize))
    self.printItemPerCategory()
    
  def getSingleImg(self,idx):
    '''
    Read a single image corresponding to the index from the dataset.
    '''
    imgFName      = os.path.join(self.dsDir,self.imgFNames[idx]+'.jpg')
    #print ('Reading image '+imgFName)
    img           = cv2.imread(imgFName,-1)
    return img
  
  def getSingleAnnotation(self,idx):
    '''
    Read and prepare the annotations corresponding to the index from the dataset.
    '''
    annFName      = os.path.join(self.annotDir,self.imgFNames[idx]+'.json')
    with open(annFName) as f:
      ann         = json.load(f)
    bBoxes          = []
    categories      = []
    categoriesName  = []
    for lbl in ann['labels']: #ann['labels'] is a list containing one or more lists; lbl is a dict
      bBox          = [];
      bBoxC0        = (lbl['box2d']['x1'],lbl['box2d']['y1'])   #tuple required for drawing rectangles in opencv
      bBoxC1        = (lbl['box2d']['x2'],lbl['box2d']['y2'])   #tuple required for drawing rectangles in opencv
      bBox.append(bBoxC0)
      bBox.append(bBoxC1)
      bBoxes.append(bBox)
      categoryIdx   = self.categories.index(lbl['category'])
      categories.append(categoryIdx)
      categoriesName.append(lbl['category'])
    #print (categories)
    #print (categoriesName)
    #print (bBoxes)
    return categories,categoriesName,bBoxes
  
  def drawBoundingBoxes(self,img,categoriesName,bBoxes):
    '''
    Get the image, bounding box coordinates and the name of the objects corresponding to
    bounding boxes as input and draw the annotations on the image.
    '''
    for bBox in bBoxes:
      idx     = bBoxes.index(bBox)    #get the index and pick the corresponding category
      cv2.rectangle(img, bBox[0], bBox[1], (0,0,255), 2)
      msg     = categoriesName[idx]
      cv2.putText(img, msg, bBox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img

  def rdSingleData(self,idx):
    '''
    Single wrapper function which reads both image and annotation corresponding to 
    an index.
    '''
    img                               = self.getSingleImg(idx)
    categories,categoriesName,bBoxes  = self.getSingleAnnotation(idx)
    return (img,categories,categoriesName,bBoxes)
    
  def rdShowAnnotatedImage(self,idx):
    '''
    Take index as an input, read the image, annotations and display the annotated image.
    '''
    img,categories,categoriesName,bBoxes  = self.rdSingleData(idx)
    img                                   = self.drawBoundingBoxes(img,categoriesName,bBoxes)
    cv2.imshow('Image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
  
  def rdData(self,strIdx,numItems):
    '''
    Take an index and the number of items as input, read the data set starting from this index
    numItems times. Returns a numpy array of all images of numItems length, category index as 
    a list of numbers and another list of list which captures the bounding box corrdinates. 
    category index and bounding box coordinates correspond to each other and are of same length.
    '''
    img                   = np.zeros(shape=(numItems,self.imgVRes,self.imgHRes,self.imgChnl),dtype=np.float32)
    categories            = []
    bBoxes                = []
    for idx in range(strIdx,strIdx+numItems):
      img[idx],c,_,bB    = self.rdSingleData(idx)
      categories.append(c)
      bBoxes.append(bB)
    #print (categories)
    #print (bBoxes)
    print (len(categories))
    print (len(bBoxes))
    
    return(img,categories,bBoxes)
    
