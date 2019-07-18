# Assignment 1 - ELL784 - Machine Learning - IIT Delhi - 2019
# Stauffer Grimson Background Subtraction

# importing libraries
from sklearn.cluster import KMeans
import numpy
import cv2
import math
import shutil   # to clean directory
import os       # to create directory


# Configuring system Parameters
videoFile = "./umcp.mpg"
framesFolder = "./frames/"
frameCapture = 1        # capture every Nth frame in video. set to higher value to reduce processor use
K = 3        # The number of gaussians assumed in the GMM. also number of cluster
alpha = 0.03 #The learning rate
T = 0.9 #minimum portion of the data that should be accounted for in the background
ro = 0.1


# Returns first value in a list. Used for sorting a list
def getFirst(t):
    return t[1]

#INITIALIZE THE MODEL
#using K means and additional approximations

def initializeModel(frame, numPixels):
    frame = frame.reshape((numPixels,1)) #reshaping the frame into a 1D array of pixels
    clt = KMeans(n_clusters = K)
    clt.fit(frame)
    print ("K means has converged after iterations ", clt.n_iter_)
    inertia = clt.inertia_
    u = clt.cluster_centers_
    r = clt.labels_
    
    si = numpy.zeros(K)
    for j in range (0,K):
        si[j] = (inertia/numPixels) #just approximation sincescikit does not seem to give variance
    
   
    #transform the r matrix as matrix with boolean values across K dimension
    rprime=numpy.zeros((numPixels,K))
    for j in range (0,K):
        for pixel in range (0,numPixels):
            if r[pixel] == j:
                rprime[pixel][j]=1

    sig = numpy.zeros((numPixels,K))
    omega = numpy.zeros((numPixels,K))
    mean = numpy.zeros((numPixels,K))

    for j in range (0,K):
         for pixel in range (0, numPixels):
            mean[pixel][j] = u[j]
            sig[pixel][j] = si[j]
            omega[pixel][j]=(1/K)*(1-alpha) + alpha * rprime[pixel][j]
    
    return (sig, mean, omega )
                   
# EXTRACT FRAMES FROM VIDEO
# open video and extract greyscale frames from video
def extractFrames(videoCapture, numPixels):

    # Cleaning up environment before processing
    # deleting the frames folder and creating folder again
    if os.path.isdir(framesFolder):
        shutil.rmtree(framesFolder)
    os.makedirs(framesFolder)

    frameId = videoCapture.get(1)       # The ID of the current frame we are on
    ret, frame = videoCapture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   
    
    
    # initialize K means clustering for first frame
    if (frameId == 0):
        (sig, mean, omega ) = initializeModel(frame,numPixels)

    #initializing background and foreground as zero    
    background = foreground = numpy.zeros(numPixels, dtype = numpy.uint8)


    #frame processing
    while(videoCapture.isOpened()):
        ret, frame = videoCapture.read()
        if (ret != True):
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)# converting to greyscale
        frameId = frameId + 1

        if (frameId % math.floor(frameCapture) == 0): # this is done just so in case we do not want to process every frame
            #Updating GMM for each pixel in frame
            (background, foreground) = processFrame(frame, sig, omega, mean, background, foreground, numPixels)
                       
            # save the foreground and background frames
            print ("Writing processed frame %d" % frameId)
            background = background.reshape(videoHeight,videoWidth)
            foreground = foreground.reshape(videoHeight,videoWidth)
            cv2.imwrite(r"%s/foreground_%s.jpg" % (framesFolder, str(int(frameId)).zfill(4)), foreground)
            cv2.imwrite(r"%s/background_%s.jpg" % (framesFolder, str(int(frameId)).zfill(4)), background)
           
    #Return the total number of frames in the video
    return frameId

# GMM on each frame video
def processFrame(frame, sig, omega, mean, background, foreground,numPixels):
    ratio = [0 for i in range(K)]
    frame = frame.reshape((numPixels,1)) # reshaping the frame into a 1D array of pixels
    background = background.reshape((numPixels,1))
    foreground = foreground.reshape((numPixels,1))



    # For each pixel in the frame compute new mean and variance
    for pixel in range (0,numPixels):
       
        gaussianMatched = 0
        sumOmega = 0
        for j in range (0,K):
            if abs(frame[pixel] - mean[pixel][j]) < (2.5 * (sig[pixel][j]) ** (0.50)):        
                mean[pixel][j] = (1 - ro) * mean[pixel][j] + ro * frame[pixel]
                sig[pixel] = (1 - ro) * sig[pixel] + ro * (frame[pixel] - mean[pixel][j]) ** 2
                omega[pixel][j] = (1 - alpha) * omega[pixel][j] + alpha
                gaussianMatched = 1
            else:
                omega[pixel][j] = (1 - alpha) * omega[pixel][j]
            sumOmega = sumOmega + omega[pixel][j]
        
        #Normalize the computed weights and find the corresponding omega/sig values          
        for j in range(0, K):
            omega[pixel][j] = omega[pixel][j] / sumOmega
            ratio[j] = omega[pixel][j] / sig[pixel][j]

        
        #sort
        (ratio,omega,mean,sig)= sort(K,ratio,omega,mean,sig,pixel)


        # If the current pixel does not belong to any gaussian, update the one with least weightage
        if gaussianMatched == 0:
            mean[pixel][K - 1] = frame[pixel]
            sig[pixel][K - 1] = 9999
            #omega[pixel][K - 1] = 0.000001

        # Check if the current pixel belongs to background or foreground
        sumOmega = 0
        B = 0
        for j in range(0, K):
            sumOmega = sumOmega + omega[pixel][j]
            if sumOmega > T:
                B = j
                break

        # Update the value of foreground and background pixel
        for j in range(0, B + 1 ):
            if gaussianMatched == 0 or abs(frame[pixel] - mean[pixel][j]) > (2.5 * (sig[pixel][j]) ** (0.50)):
                foreground[pixel] = frame[pixel]
                background[pixel] = mean[pixel][j]
                break
            else:
                foreground[pixel]= 255
                background[pixel] = frame[pixel]
                
    return (background, foreground)    

def sort(K,ratio,omega,mean,sig,pixel):
    # Arrange the mean, variance and weights in decreasing order as per the ratio omega/sig
    # create a list (called records) with all the the values to be sorted
    records =[]
    for j in range(0, K):
        records.append(( ratio[j], omega[pixel][j], mean[pixel][j],sig[pixel][j] ))
    # Now sort by first key i.e. omega/sig
    records.sort(key = getFirst ,reverse=True)
        
     # get the values back from list and putting back into the variables
    for j in range(0,K):
        (ratio[j],omega[pixel][j],mean[pixel][j],sig[pixel][j])=records[j]
            #print("sorted records ",j," is now", records[j])
                                 
    return(ratio,omega,mean,sig)
             

# COMPILE FRAMES BACK TO VIDEO
def compileVideo(nFrames, videoWidth, videoHeight,filename):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    print ("Trying to compile ", filename, "file")
    outputVideo = cv2.VideoWriter(r"%s.avi" % (filename), fourcc, 30.0, (videoWidth, videoHeight))
    # Write the frames into the video
    for counter in range (0, nFrames-1):
        print("processing frame",counter)
        # Read the back frame
        Frame = cv2.imread(r"%s/%s_%s.jpg" % (framesFolder, filename,str(int(counter)).zfill(4)), 1)
        # Write the frame in to the video
        outputVideo.write(Frame)
    outputVideo.release()
    
# MAIN ENTRY PROGRAM
if __name__ == "__main__":

    videoCapture = cv2.VideoCapture(videoFile)

    # Understanding Video Propertirs
    videoWidth = round( videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) )
    videoHeight = round( videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    frameRate = round( videoCapture.get(cv2.CAP_PROP_FPS) )
    numPixels = videoWidth * videoHeight

    #extract frames from the video and process it to k means andupdate background and foreground
    numFrames = extractFrames(videoCapture, numPixels)
   
    #frames to video
    compileVideo(int(numFrames), videoWidth, videoHeight,"foreground")
    compileVideo(int(numFrames), videoWidth, videoHeight,"background")
    
    # cleaning up
    print("Cleaning Up")
    videoCapture.release()
    cv2.destroyAllWindows()

