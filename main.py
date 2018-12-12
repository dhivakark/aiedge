from supportLib import *

#globals
#DATASET_DIR       = 'D:\\general\\ML_DL\\AIEdgeContestObjectDetection_Signate\\dataset'
DATASET_DIR       = os.path.join ('../','dataset')
CATEGORY_LST      = ('Bicycle', 'Bus', 'Car', 'Motorbike', 'Pedestrian',
                     'SVehicle', 'Signal', 'Signs', 'Train', 'Truck')

if __name__ == '__main__' :
  trainImgPath    = os.path.join(DATASET_DIR, 'dtc_train_images')
  trainAnnPath    = os.path.join(DATASET_DIR, 'dtc_train_annotations')
  
  trainSet        = dataset(trainImgPath, trainAnnPath, CATEGORY_LST)  
  
  #Plot and see the distribution of data
  #trainSet.visualizeDsDistribution()
  #trainSet.printDataSetStats()
  
  #Show the annotated image by selecting some samples
  #trainSet.rdShowAnnotatedImage(1)
  
  #Read the data set in batches, with startind index and number of items
  trainSet.rdData(strIdx=0,numItems=1000)
