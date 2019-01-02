#imports
from os import listdir
from os.path import isfile, join
import os
import numpy as np
from PIL import Image
from skimage import transform
from random import shuffle
import skimage.io as skio
from skimage import util
from skimage import io
from scipy import ndarray
import random
import skimage as sk
from matplotlib.pyplot import imshow, pause, show
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
#class
class imageDataGenerator:
    images = []
    masks = []    
    imagesPath = []
    masksPath = []   
    index = 0
    batchSize = 0
    resize = True
    size = (256, 256)
    model = []
    epochs = 100
    diceResult = []
    augment = True
    def __init__(self, model, augment = True, resize = True, size = (1024, 1024), plotname = 'graph.png'):
        self.batchSize = 1
        self.resize = resize
        self.size = size
        self.model = model
        self.epochs = 50
        self.plotname=plotname
        self.imagesPath = []
        self.masksPath = []
        self.index = 0
        self.diceResult = []
        self.augment = augment
        self.batchCount = 0
        self.batchCounter = 1
        self.trainLossPlot = []
        self.testLossPlot = []
        self.testImagesPath = []
        self.testMasksPath = []
    def getPaths(self, paths):
        #gets paths of all images in input folder and saves them to global variables
        self.imagesPath = []
        self.masksPath = []
        for pathToImages in paths:
            fileNames = [f for f in listdir(pathToImages) if isfile(join(pathToImages, f))]
            
            borderslices = {'Ascl_1mut_CreERT2_E165': (155, 886),
                 'Ascl-CreERT-E165-ctrl': (82, 678),
                 'Red_Del(-90-cz)xInv(-33-cz)_24783e5': (200, 959),
                 'Green_Inv(-500)xInv(-500)_25377e4': (88, 830),
                 'Green_WT_25280e6': (262, 942),
                 'Blue_12761-1g++_Chd7g+6+Chd': (182, 835),
                 'Green_25280e5': (249, 990)}
            curSample = os.path.basename(os.path.normpath(pathToImages))
            rng = borderslices[curSample]
            tempMask = []
            tempIm = []
            for imName in fileNames:
                if  '_mask' in imName.lower():
                    tempMask.append(join(pathToImages,imName))
                elif imName == 'Thumbs.db':
                    imName = None
                else:
                    tempIm.append(join(pathToImages,imName))
            self.imagesPath.extend(tempIm[:rng[0]][0::10])
            self.imagesPath.extend(tempIm[rng[0]:rng[1]][0::3])
            self.imagesPath.extend(tempIm[rng[1]:][0::10])
            self.masksPath.extend(tempMask[:rng[0]][0::10])
            self.masksPath.extend(tempMask[rng[0]:rng[1]][0::3])
            self.masksPath.extend(tempMask[rng[1]:][0::10])
            #self.imagesPath.extend(tempIm)
            #self.masksPath.extend(tempMask)
        #self.imagesPath = self.imagesPath[0::3]
        #self.masksPath = self.masksPath[0::3]
        self.batchCount = ceil(len(self.imagesPath)/self.batchSize)
    def shuffleData(self):
        #shuffles image paths in global variables
        joined = list(zip(self.imagesPath, self.masksPath))
        shuffle(joined)
        self.imagesPath, self.masksPath = zip(*joined)
        self.imagesPath = list(self.imagesPath)
        self.masksPath = list(self.masksPath)
        
    def loadBatch(self, imagesPath, masksPath, batchSize, test=False):
        #loads batch from data folder
        if test:
            if self.resize:
                self.masks = np.expand_dims(np.array([transform.resize(np.array(Image.open(fname)), self.size, preserve_range=True, order= 0) for fname in masksPath[self.index:self.index+batchSize]]), axis = 3)
                self.images = np.expand_dims(np.array([transform.resize(np.array(Image.open(fname)), self.size, preserve_range=True, order = 0) for fname in imagesPath[self.index:self.index+batchSize]]), axis = 3)
            else:
                self.masks = np.expand_dims(np.array([np.array(Image.open(fname)) for fname in masksPath[self.index:self.index+batchSize]]), axis=3)
                self.images = np.expand_dims(np.array([np.array(Image.open(fname)) for fname in imagesPath[self.index:self.index+batchSize]]), axis = 3)
            self.index = self.index + batchSize
            self.batchCounter += 1
            return(None)
        if self.batchCounter < self.batchCount:
            if self.resize:
                self.masks = np.expand_dims(np.array([transform.resize(np.array(Image.open(fname)), self.size, preserve_range=True, order= 0) for fname in masksPath[self.index:self.index+batchSize]]), axis = 3)
                self.images = np.expand_dims(np.array([transform.resize(np.array(Image.open(fname)), self.size, preserve_range=True, order = 0) for fname in imagesPath[self.index:self.index+batchSize]]), axis = 3)
#            self.masks [self.masks > 0.5] = 1
#            self.masks [self.masks <= 0.5] = 0
            else:
                self.masks = np.expand_dims(np.array([np.array(Image.open(fname)) for fname in masksPath[self.index:self.index+batchSize]]), axis=3)
                self.images = np.expand_dims(np.array([np.array(Image.open(fname)) for fname in imagesPath[self.index:self.index+batchSize]]), axis = 3)
        else:
            finalBatchSize = len(imagesPath)%batchSize
            if finalBatchSize == 0:
                finalBatchSize = self.batchSize
            if self.resize:
                self.masks = np.expand_dims(np.array([transform.resize(np.array(Image.open(fname)), self.size, preserve_range=True, order= 0) for fname in masksPath[self.index:self.index+finalBatchSize]]), axis = 3)
                self.images = np.expand_dims(np.array([transform.resize(np.array(Image.open(fname)), self.size, preserve_range=True, order = 0) for fname in imagesPath[self.index:self.index+finalBatchSize]]), axis = 3)
#            self.masks [self.masks > 0.5] = 1
#            self.masks [self.masks <= 0.5] = 0
            else:
                self.masks = np.expand_dims(np.array([np.array(Image.open(fname)) for fname in masksPath[self.index:self.index+finalBatchSize]]), axis=3)
                self.images = np.expand_dims(np.array([np.array(Image.open(fname)) for fname in imagesPath[self.index:self.index+finalBatchSize]]), axis = 3)
            
        self.index = self.index + batchSize
        self.batchCounter += 1
    def trainModel(self, pathsToData, epochs, batchSize, pathsToTestData = False):
        self.epochs = epochs
        self.batchSize = batchSize
        #trains model
        self.getPaths(pathsToData)
        if pathsToTestData != False:
            self.getTestPath(pathsToTestData)
        self.index = 0
        for i in range(0, self.epochs):
            self.shuffleData()
            batchIndex = 0
            self.batchCounter = 1
            lossInEpoch = []
            while (batchIndex * self.batchSize) < len(self.imagesPath):
                self.loadBatch(self.imagesPath, self.masksPath, self.batchSize)
                #self.normalizeImages()
                
#                numOfCartilagePixels = np.sum(self.masks)
#                
#                if numOfCartilagePixels == 0:
#                    print('Skip')
#                    batchIndex += 1
#                    continue
                loss = self.model.train_on_batch(self.images, self.masks)
                lossInEpoch.append(loss[1])
                print ('Epoch: ', str(i+1) , ' ', 'Batch: ', str(batchIndex + 1), ', ', self.model.metrics_names[0], "=", loss[0], "-", self.model.metrics_names[1], "=", loss[1])
                batchIndex += 1
                                           
                #augmentations
                if self.augment == True:
                    #augmentations = ['noise', 'rotate', 'flip', 'rotatenoise', 'flipnoise', 'fliprotate', 'fliprotatenoise', 'none']
                    #elasticke transformace, zatim asi nefunkcni
                    #augmentations = ['rotate', 'flip', 'fliprotate', 'elastic', 'elasticrotate', 'elasticflip', 'elasticfliprotate', 'none']
                    #probabilities = [0.06, 0.12, 0.09, 0.09, 0.09, 0.09, 0.09, 0.37]
                    #aktualne funkcni transformace
                    #augmentations = ['flip', 'none']
                    #probabilities = [0.42, 0.58]
                    augmentations = ['rotate', 'flip', 'fliprotate', 'none']
                    probabilities = [0.14, 0.14, 0.14, 0.58]
                    #probabilities = [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.58]
                    augImBatch = []
                    augMaskBatch = []
                    for k in range(0, len(self.images)):
                        imageToAugment = np.squeeze(self.images[k])
                        maskToAugment = np.squeeze(self.masks[k])
                        
#                        augim, augmask = self.horizontal_flip(imageToAugment, maskToAugment)
#                        loss = self.model.train_on_batch(np.expand_dims(np.expand_dims(augim, axis = 3), axis= 0), np.expand_dims(np.expand_dims(augmask, axis = 3), axis =0))
#                        print('Flip --- Epoch: ', str(i+1) , ' ', 'Batch: ', str(batchIndex + 1), ', ', self.model.metrics_names[0], "=", loss[0], "-", self.model.metrics_names[1], "=", loss[1])
                        
                        aug = np.random.choice(augmentations, 2, p = probabilities)
                        for transformation in aug:
                            if transformation == 'noise':
                                augim, augmask = self.random_noise(imageToAugment, maskToAugment)
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'rotate':
                                augim, augmask = self.random_rotation(imageToAugment, maskToAugment)
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'rotatenoise':
                                augim, augmask = self.random_noise(imageToAugment, maskToAugment)
                                augim, augmask = self.random_rotation(augim, augmask)
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'flip':
                                augim, augmask = self.horizontal_flip(imageToAugment, maskToAugment)
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'flipnoise':
                                augim, augmask = self.horizontal_flip(imageToAugment, maskToAugment) 
                                augim, augmask = self.random_noise(augim, augmask)
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)                                                                                      
                            elif transformation == 'fliprotate':
                                augim, augmask = self.horizontal_flip(imageToAugment, maskToAugment) 
                                augim, augmask = self.random_rotation(augim, augmask) 
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'fliprotatenoise':
                                augim, augmask = self.horizontal_flip(imageToAugment, maskToAugment) 
                                augim, augmask = self.random_rotation(augim, augmask) 
                                augim, augmask = self.random_noise(augim, augmask)
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'elastic':
                                augim, augmask = self.elastic_transform(imageToAugment, maskToAugment, 512, 16)
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'elasticrotate':
                                augim, augmask = self.elastic_transform(imageToAugment, maskToAugment, 512, 16)
                                augim, augmask = self.random_rotation(augim, augmask) 
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'elasticflip':
                                augim, augmask = self.horizontal_flip(imageToAugment, maskToAugment) 
                                augim, augmask = self.elastic_transform(augim, augmask, 512, 16)
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            elif transformation == 'elasticfliprotate':
                                augim, augmask = self.horizontal_flip(imageToAugment, maskToAugment) 
                                augim, augmask = self.elastic_transform(augim, augmask, 512, 16)
                                augim, augmask = self.random_rotation(augim, augmask) 
                                augImBatch.append(augim)
                                augMaskBatch.append(augmask)
                            if len(augImBatch) == self.batchSize:
                                loss = self.model.train_on_batch(np.expand_dims(augImBatch, axis = 3), np.expand_dims(augMaskBatch, axis = 3))
                                lossInEpoch.append(loss[1])
                                print('Augmented --- Epoch: ', str(i+1) , ' ', 'Batch: ', str(batchIndex), ', ', self.model.metrics_names[0], '=', loss[0], '-', self.model.metrics_names[1], '=', loss[1])
                                augImBatch = []
                                augMaskBatch = []
                          
            self.index = 0
            self.trainLossPlot.append(sum(lossInEpoch)/len(lossInEpoch))
            print('Average loss in epoch ', str(i+1),': ', str(sum(lossInEpoch)/len(lossInEpoch)))
            
            if pathsToTestData != False:
                testDiceInEpoch = []
                for j in range(0, len(self.testImagesPath)):
                    self.loadBatch(self.testImagesPath, self.testMasksPath, 1, test=True)
                    #self.normalizeImages
                    testDiceInEpoch.append(self.model.test_on_batch(self.images, self.masks)[1])
                self.testLossPlot.append(sum(testDiceInEpoch)/len(testDiceInEpoch))
                print('Average test loss in epoch ', str(i+1),': ', str(sum(testDiceInEpoch)/len(testDiceInEpoch)))
                self.index = 0
        self.make_plot(np.linspace(1, i+1,i+1), self.trainLossPlot, self.testLossPlot)

    def saveModel(self, name):
        #saves trained model
        self.model.save(name)
    def saveWeights(self, name):
        #saves trained weights
        self.model.save_weights(name)
    def getTestPath(self, pathToData):
        #gets paths to test data
        self.testImagesPath = []
        self.testMasksPath = []
        for pathToImages in pathToData:
            fileNames = [f for f in listdir(pathToImages) if isfile(join(pathToImages, f))]
        
            for imName in fileNames:
                if  '_mask' in imName.lower():
                    self.testMasksPath.append(join(pathToImages,imName))
                elif imName == 'Thumbs.db':
                    imName = None
                else:
                    self.testImagesPath.append(join(pathToImages,imName))
    def predictMasks(self, pathToData, savePath):
        #uses loaded model to predict masks of input test images
        self.getTestPath(pathToData)
        self.index = 0
        batchIndex = 0
        resultIndex = 0
        self.diceResult = []
        while batchIndex < len(self.testImagesPath):
            self.loadBatch(self.testImagesPath, self.testMasksPath,1, test = True)
            #self.normalizeImages
            #self.diceResult.append(self.model.test_on_batch(self.images, self.masks))
            results = self.model.predict_on_batch(self.images)
            results = np.squeeze(results)
            skio.imsave(join(savePath,"%d_predict.png"%resultIndex),results)
            resultIndex += 1
            batchIndex += 1
            
    def random_rotation(self, image_array, mask_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-25, 25)
        maximum = np.max(np.abs(image_array))
        image_array = image_array/maximum
        rotmask = sk.transform.rotate(mask_array, random_degree, order = 0)
        rotmask[rotmask>0] = 1
        return sk.transform.rotate(image_array, random_degree, order = 0)*maximum, rotmask

    def random_noise(self, image_array, mask_array):
    # add random noise to the image
        maximum = np.max(np.abs(image_array))
        image_array = image_array/maximum
        return sk.util.random_noise(image_array)*maximum, mask_array

    def horizontal_flip(self, image_array: ndarray, mask_array : ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
        return image_array[:, ::-1], mask_array[:, ::-1]
    
    def normalizeImages(self):
        #image normalization
        self.images = (self.images - self.images.min()) / (self.images.max() - self.images.min())
        self.images = (self.images - np.mean(self.images))/np.std(self.images)
    def make_plot(self, epochs, trainingAccuracy, testingAccuracy):
        #epochs/accuracy plot for test and train data
        plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        line1, = plt.plot(epochs,trainingAccuracy, label = 'Train')
        line2, = plt.plot(epochs,testingAccuracy, label = 'Test')
        plt.ylim([0, 1]) 
        first_legend = plt.legend(handles=[line1], loc=4)
        ax = plt.gca().add_artist(first_legend)
        plt.legend(handles=[line2], loc=3)
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy')
        plt.savefig('grafy\\'+self.plotname)
        
    def elastic_transform(self, image, mask, alpha, sigma, random_state=None):
        assert len(image.shape)==2
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))    
        return map_coordinates(image, indices, order=0).reshape(shape), map_coordinates(mask, indices, order=0).reshape(shape)