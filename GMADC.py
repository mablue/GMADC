#=============================================================================#
#    GMADC Algorythm(Its main file to run)                                    #
#    To solve random choise on disconected Softwere class graps.              #
#    By: Masoud Azizi     Email: mablue92@gmail.com                           #
#=============================================================================#

import ntpath
import os
import random
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from matplotlib import pyplot as plt
from numpy import inf
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
# from progress.bar import Bar
# for drawing plot
from callgraph import CallGraph
#My text mining algorithm to classify disconnected classes in GMA
from GMADCC import GMADisconnectedClassClassifier as gmadc
newGmadc=None
# To prevent from loops in chain that will make by text similarity
disconnectedClassesIndex=list()
# disSimList: disconnected classes+Similar Classes that obtained form GMADCC (ex: disSimList=[[3,4],[1,3],[0,2]])
disSimList=list()

# disconnected classes index list
dccil = list()
dcSim=list()
dcAndSim = list()
filePaths=None


# search in 2d arrays by value
def index_2d(data, search):
    for i, e in enumerate(data):
        try:
            return i, e.index(search)
        except ValueError:
            pass
    raise ValueError("{} is not in list".format(repr(search)))


# create Similarity Matrix
def createSimilarityMatrix(shortestCallMatrix):
    # print("createSimilarityMatrix(shortestCallMatrix)")
    tempMatrix = [[0 for i in range (len (shortestCallMatrix))] for i in range (len (shortestCallMatrix))]
    v = len (tempMatrix)
    
    maximum = -1    
    for i in range(0,v):
        max_temp = max(shortestCallMatrix[i])
        if maximum < max_temp:
            maximum = max_temp

    if maximum == 0:
        print("all classes are in its own cluster!")
        exit()
    
    for i in range (0, v):
        for j in range (0, v):
            if shortestCallMatrix[i][j] == 0 and i !=j:
                tempMatrix[i][j] = maximum
            else:
                tempMatrix[i][j] = shortestCallMatrix[i][j]
        
    for i in range (0,v):
        for j in range (0, v):
            tempMatrix[i][j] = 1 - tempMatrix[i][j]/maximum
    return tempMatrix


# Initializing
def initializing(k , n, similarityMatrix):
    # print("initializing")

    centers = generateRandomCenter(k, n)
    #print(centers)    
    
    clusters = fillClustersBasedOnCenter(k, n, similarityMatrix, centers)
    
    return clusters


# Fill Clusters Based On Center
def fillClustersBasedOnCenter(k, n, similarityMatrix, centers):
    # print("fillClustersBasedOnCenter(k, n, similarityMatrix, centers)")
    disconnectedClassesIndex=list()
    
    for i in range(n): # initialize base step, center of a cluster remains in index 0 of each list
        index = 0
        clusterNumber = centers[0][0]
        maxSimilarity = similarityMatrix[i][clusterNumber]
        for j in range(1,k):
            clusterNumber = centers[j][0]
            ###################################################
            #############,               (#####################
            #########                         #################
            ######.            *#%#,            %##############
            ####/        %###############         #############
            ###        #####################        ###########
            ##       #########################       ##########
            #&      ##########################.      ##########
            #       ###########################      ##########
            #,      ###########################      &#########
            ##       #########################       ##########
            ##&       #######################       ###########
            ####        ###################        ############
            #####          ############*            ###########
            ########                                  /########
            ###########                     #            ######
            #################         #########            ,###
            ######################################            #
            ########################################          #
            ###########################################    ####
            ###################################################

            if maxSimilarity < similarityMatrix[i][clusterNumber]:
                maxSimilarity = similarityMatrix[i][clusterNumber]
                index = j
                # print("<",i,j)
            # Place that we reach to a disconnected class and MA_MS algo starts his text mining calculation to find a
            # similarity between other classes and make a chain with other clusters and disconnected classes
            elif maxSimilarity == similarityMatrix[i][clusterNumber] and j==1:
                # disconnectedClassesIndex.append(i)
                # if disconnectedClassesIndex.count(i)==2:  
                if i not in dccil:
                    dccil.append(i)                
                    dcAndSim.append([i,i])
                # print(dccil)
                
                # print(dccil)
                # print(clusterNumber)

                # disconnectedClassesIndex.append(i)
                # if disconnectedClassesIndex.count(i)==2:                
                #     print("---------------------------------------------")                        
                #     dcsi = newGmadc.getDisconnectedClassCenter(disconnectedClassesIndex,filePaths,clusterNumber)
                #     if disSimList.count((i,dcsi))<1: 
                #         disSimList.append((i,dcsi))
            
                # print(disconnectedClassesIndex)

                

                # pass

            # elif maxSimilarity > similarityMatrix[i][clusterNumber]:
                # print(">",i,j)
                # pass

            # print("filePath(i={},j={})={},MaxSim={},SimMat[{}][{}]={}".format(i,j,filePaths[i],maxSimilarity,i,clusterNumber,similarityMatrix[i][clusterNumber]))

        temp = []
        temp = centers[index]
        if i != temp[0]:
            temp.append(i)

        centers.pop(index)  
        centers.insert(index,temp)

           
        # print("i:", i, "index:", index, "sim:", maxSimilarity, "centers:", centers)
    return centers


# Generate Random Center
def generateRandomCenter(k, n):
    # print("generateRandomCenter(k, n)")
    clusters = []
    
    center = random.sample(range(n), k) #create k random center of k-means
    #print(center)
    
    for j in range(k):  #transform k centers to 2 dimentionals array
        c = center.pop(0)  
        clusters.append([c])

    del(center) 
    #print(clusters)
    return clusters
    
# Copy
def copy(matrix):
    # print("copy(matrix)")
    l = len(matrix)
    temp = []

    for i in range(0,l):
        temp.append([])
        m = len(matrix[i])
        for j in range(0,m):
            temp[i].append(matrix[i][j])
    return temp
    
# Correct Center   
def correctCenter(clusters, similarityMatrix):
    # print("correctCenter(clusters, similarityMatrix)")
    clustersUpdateCenter = []
    for centerIndex in range(k):
        clustersBackup = copy(clusters)
        popedCluster = clustersBackup.pop(centerIndex)
        # print("popedCluster:", popedCluster)
        
        
        sameSimIndex = [0]
        maxSimilarity = 0
        
        for i in range(0, len(popedCluster)):
            
            popedClusterSimilaritySum = 0    
            for j in range(0, len(popedCluster)):
                popedClusterSimilaritySum = popedClusterSimilaritySum + similarityMatrix[popedCluster[i]][popedCluster[j]]
            # print(popedCluster[i], ":" , popedClusterSimilaritySum)
            if popedClusterSimilaritySum > maxSimilarity:
                maxSimilarity = popedClusterSimilaritySum
                sameSimIndex.clear()        
                sameSimIndex.append(i)
            elif popedClusterSimilaritySum == maxSimilarity:
                sameSimIndex.append(i)
                # print(i)
                # print(sameSimIndex)

        
        # print("sameSimIndex: ", sameSimIndex, "maxSimilarity: ",maxSimilarity)
        
        if len(sameSimIndex) == 1:
            val = popedCluster[sameSimIndex[0]]
            # print(popedCluster)    
            popedCluster.remove(val) 
            # print(popedCluster)
            popedCluster.insert(0, val)
            # print(popedCluster)    
            clustersUpdateCenter.append(popedCluster)
            # print("OK, clustersUpdateCenter: ",clustersUpdateCenter,"\n")
        
        elif len(sameSimIndex) > 1:
            tempMinOtherSim = n    
            for i in range(len(sameSimIndex)):
                sumOtherSimilarity = 0       
                val = popedCluster[sameSimIndex[i]]
                for j in range(k-1):
                    for m in range(len(clustersBackup[j])):
                        sumOtherSimilarity = sumOtherSimilarity + similarityMatrix[val][clustersBackup[j][m]]
                # print("val:" ,val, "sumOtherSimilarity:",sumOtherSimilarity)
                if tempMinOtherSim > sumOtherSimilarity:
                    tempMinOtherSim = sumOtherSimilarity
                    tempVal = val
            # print("tempVal:", tempVal)
            popedCluster.remove(tempVal)
            popedCluster.insert(0, tempVal)
            clustersUpdateCenter.append(popedCluster)
            # print("NOK, clustersUpdateCenter: ",clustersUpdateCenter,"\n")
    return clustersUpdateCenter

# Compute Similarity Function
def computeSimilarityFunction(matrix , similarityMatrix):#sum of similarity of all clusters 
    # print("computeSimilarityFunction(matrix , similarityMatrix)")
    function = 0
    for i in range(k):
        for j in range(1, len(matrix[i])):
            function = function + similarityMatrix[matrix[i][0]][matrix[i][j]]
    return function

# Clustering
def clustering(k, n, similarityMatrix, clustersUpdateCenter):
    # print("clustering(k, n, similarityMatrix, clustersUpdateCenter)")
    clusterOld = copy(clustersUpdateCenter) 

    iteration = 0
    flag = 0

    while iteration <1000 and flag < 5:

        iteration = iteration + 1
        clusterNew = []
        
        # print("old:", id(clusterOld), "new:", id(clusterNew) , "d:", id(clusterOld)-id(clusterNew),"\n")    
        
        for i in range(k):
            clusterNew.append([clusterOld[i][0]]) 
        
        # print(centerUpdate)
        clusterNew = fillClustersBasedOnCenter(k, n, similarityMatrix, clusterNew)
        # print("clusterNew: ",clusterNew)
        
        # similarityFunctionUpdate = computeSimilarityFunction(clusterNew, similarityMatrix)  
        # print("similarityFunctionUpdate: ",similarityFunctionUpdate,"\n")
        
        
        clusterNew = correctCenter(clusterNew, similarityMatrix)
        # print("clustersUpdateCenter: ",clusterNew)
        
        # similarityFunctionUpdate = computeSimilarityFunction(clusterNew, similarityMatrix)  
        # print("similarityFunctionUpdate: ",similarityFunctionUpdate,"\n")
        
        # check for continuing    
        tempNew = copy(clusterNew)
        tempOld = copy(clusterOld)
        # print(id(tempNew),id(tempOld),id(clusterNew),id(clusterOld))
        for i in range(len(tempNew)):
            tempNew[i].sort()
        tempNew.sort()   
        # print("Sorted New: ", tempNew)
    
        for i in range(len(tempOld)):
            tempOld[i].sort()
        tempOld.sort()   
        # print("Sorted Old: ", tempOld)
        # print()
        
        if(tempNew == tempOld):
            flag = flag + 1
    
        
        del(tempNew)
        del(tempOld)
    
        clusterOld = copy(clusterNew) 
    
    return clusterNew
   
# Unused
# Exporting To MoJo Format Algorithm Manual
def exportingToMoJoFormatAlgorithmManual(k, n, clustersFinal):
    # print("exportingToMoJoFormatAlgorithmManual(k, n, clustersFinal)")
    for i in range(k):
        clustersFinal[i].sort()
    clustersFinal.sort()
    f1 = open ("MoJoAlgorithmManual.txt", "w")
    for centerIndex in range(k):
        for i in range(len(clustersFinal[centerIndex])):
            f1.write("contain ")
            f1.write("hulu")
            f1.write(str(centerIndex))
            f1.write(" ")
            f1.write(str(clustersFinal[centerIndex][i]))
            f1.write("\n")
    f1.close()
    

# exportingToMoJoFormatExpert
def exportingToMoJoFormatExpert(fileNames, filePathNames, folderPathResult):
    # print("exportingToMoJoFormatExpert(fileNames, filePathNames, folderPathResult)")
    temp = []

    for i in range(len(fileNames)):
        # temp[i][1] = fileNames[i]
        indexCluster = findInTofilePtheNames (fileNames[i], filePathNames)
        
        if indexCluster == -1:
            print("Cluster Conflict class ", fileNames[i])
            return
        
        # temp[i][0] = filePathNames[indexCluster][0]
        temp.append([filePathNames[indexCluster][0], fileNames[i]])
    temp.sort()
    f1 = open (folderPathResult + "/MoJoExpert.txt", "w")
    for i in range(len(fileNames)):
        f1.write("contain ")
        f1.write(temp[i][0])
        f1.write(" ")
        f1.write(temp[i][1])
        f1.write("\n")
    f1.close()
    del(temp)
    

    
# findInTofilePtheNames
def findInTofilePtheNames (className, filePathNames):
    # print("findInTofilePtheNames (className, filePathNames)")
    index = -1
    for i in range(len(filePathNames)):
        if className == filePathNames[i][1] or className == filePathNames[i][2]:
            if index == -1:
                index = i
            else:
                return -1
    return index
        
# findInTofilePtheNames
def exportingToMoJoFormatAlgorithm(k, n, clustersFinal, fileNames, filePathNames, run_no, folderPathResult):
    # print("exportingToMoJoFormatAlgorithm(k, n, clustersFinal, fileNames, filePathNames, run_no, folderPathResult)")
    f1 = open (folderPathResult + "/MoJoAlgorithm" + str(k) + "_" + str(run_no) + ".txt" , "w")
    for centerIndex in range(k):
        for i in range(len(clustersFinal[centerIndex])):
            f1.write("contain ")
            f1.write("hulu")
            f1.write(str(centerIndex))
            f1.write(" ")
            f1.write(fileNames[clustersFinal[centerIndex][i]])
            f1.write("\n")
    f1.close()

# Start
cvsFilePathString = []

pathResultString = []
# print("for root, dirs, files in os.walk(\"CaseStudies\")")
for root, dirs, files in os.walk("CaseStudies"):
    for file in files:
        if file.endswith(('.csv')):
            cvsFilePath=os.path.join(root, file)
            cvsFilePathString.append(cvsFilePath)

            resultPath = os.path.join(root,"../result",file)
            pathResultString.append(resultPath)
            if not os.path.isdir(resultPath):
                os.makedirs(resultPath)

# print("for i in range (len(pathResultString))")
for i in range (len(pathResultString)):
    if not os.path.isdir(pathResultString[i]):
        os.mkdir(pathResultString[i])


# print("for cvsFileNumber in range (0,len(cvsFilePathString))")
for cvsFileNumber in range (0,len(cvsFilePathString)):
    dcSim=list()
    cvsFp=cvsFilePathString[cvsFileNumber]
    df = pd.read_csv(cvsFp)
    df.fillna(0, inplace = True) 

    sourceCodeFp="SourceCodes/{}.src/".format(ntpath.basename(cvsFp)[0:-4])
    filePaths= [sourceCodeFp+n for n in list(df.columns.values[1:])]


    newGmadc = gmadc(filePaths)





    fileDirs=["'{}'".format(os.path.dirname(path)).replace(" ","%20") for path in filePaths]

    fileNames=list(["'{}'".format(ntpath.basename(path)) for path in filePaths])

    fileExts=list([os.path.splitext(name)[1] for name in fileNames])

    df.drop(df.columns[0], axis=1, inplace=True)
    cdgNonSquer=df.to_numpy()
    cdgNonSquerMaxLen=max(len(cdgNonSquer[0,:]),len(cdgNonSquer[:,0]))
    # print(cdgNonSquerMaxLen,len(cdgNonSquer[0,:]),len(cdgNonSquer[:,0]))
    cdgNonSymetric=np.resize(cdgNonSquer,(cdgNonSquerMaxLen,cdgNonSquerMaxLen))
    SymetricCDG = np.maximum( cdgNonSymetric, cdgNonSymetric.transpose() )
    # print("SymetricCDG\n",SymetricCDG)
    timeInit = 0
    timeClustering = 0
    timeTotal = 0

    filePathNames = [[fileDirs[i],fileNames[i],fileNames[i]] for i in range(len(fileNames))]

    

    # print(np.corrcoef(callMatrix))


    # print(filePathNames)
    print(cvsFilePathString[cvsFileNumber]+":\tinput files completed :)")

    exportingToMoJoFormatExpert(fileNames, filePathNames, pathResultString[cvsFileNumber])
    print(cvsFilePathString[cvsFileNumber]+":\texporting to MoJo format for expert completed :)")

    time1 = time.time()
    time2 = time.time()
    timeInit = time2 - time1
    print(cvsFilePathString[cvsFileNumber]+":\tgenerating call matrix to symmetric completed :)")

    csrCDG = csr_matrix(SymetricCDG)
    # print("csrCDG\n", csrCDG)

                                   #############   
                                      ##########   
                                      ##########   
                                    ############   
                                 #######      ##    
             ##                #######   ##        
           ######           #######   #####        
         ###########      #######   #######        
      ######## ####### #######     ########        
    #######       ##########   ##  ########        
    #####   ###     #####   #####  ########        
         ######  ###  #   #######  ########        
        #######  #######  #######  ########        
        #######  #######  #######  ########        
        #######  #######  #######  ########        
        #######  #######  #######  ########        
        #######  #######  #######  ########        
        #######  #######  #######  ########        
    # to draw the plot
    cg = CallGraph(csrCDG )
    cg.draw()


    dist_matrix = shortest_path(csgraph=csrCDG,method='FW')# FW: floydwarshal
    dist_matrix[dist_matrix == inf] = 0
    # print(dist_matrix)

    # Similarity Matrix Calculation
    similarityMatrix = cosine_similarity(dist_matrix)
    # print("for i in range(len(similarityMatrix))")
    for i in range(len(similarityMatrix)):
        similarityMatrix[i][i]=1
    # np.savetxt('cosine_similarity.csv', similarityMatrix, delimiter=',', fmt='%s')
    # print("simMtx\n",similarityMatrix)


    # similarityMatrix = createSimilarityMatrix(dist_matrix)

    time1 = time.time()
    time2 = time.time()
    timeInit = timeInit + time2 - time1
    print(cvsFilePathString[cvsFileNumber]+":\tcalculating shortest path matrix completed :)")

    time1 = time.time()

    # similarityMatrix = createSimilarityMatrix(dist_matrix)
    # np.savetxt('createSimilarityMatrix.csv', similarityMatrix, delimiter=',', fmt='%s')

    # print(np.array(similarityMatrix))

    time2 = time.time()
    timeInit = timeInit + time2 - time1
    print(cvsFilePathString[cvsFileNumber]+":\tforming similarity completed :)")

    # print(similarityMatrix)

    n = len(SymetricCDG) # number of elements
    result = []
    maxRunNo = 1  # default:  maxRunNo = 30
    maxK = 3  # default:
    maxK = int(min(int(n/3), 100))

    # bar = Bar('Processing', max=maxK*maxRunNo) 


    # Mohem
    # print("for run_no in range (1,maxRunNo+1)")
    for run_no in range (1,maxRunNo+1):
        # print("for k in range (2,maxK + 1)")
        for k in range (2,maxK + 1):
            print("Progress: {}/{},{}/{},{}/{} ".format(cvsFileNumber,len(cvsFilePathString),run_no,(maxRunNo),k,(maxK)))
            # bar.next()
            time1 = time.time()
            clustersInit = initializing(k , n, similarityMatrix)

            # print("After initializing:\nclusters: ",clustersInit)
            # print("initializing culsters completed :)")            
            # similarityFunction = computeSimilarityFunction(clustersInit, similarityMatrix)  
            # print("similarityFunction:",similarityFunction,"\n")

            clustersUpdateCenter = correctCenter(clustersInit, similarityMatrix)

            # print("clustersUpdateCenter: ",clustersUpdateCenter)
            # similarityFunctionUpdate = computeSimilarityFunction(clustersUpdateCenter, similarityMatrix)  
            # print("similarityFunctionUpdate: ",similarityFunctionUpdate)

            clustersFinal = clustering(k, n, similarityMatrix, clustersUpdateCenter)
           
           
            #####################################, ############
            #####################################     &########
            #####################################        .#####
            #,              ###########(                     ##
            #,                 ######                         #
            #############(      .##       ,######         %####
            ###  ######  ##.    ,&      *########      ########
            # *########## &##  #       ##########  %###########
            # ############ ###(      %#########################
            # *###& .#### &##       #  ##########&%############
            ###  ######  ##,      #     #########     #########
            ##############      .##       #######        &#####
            #,                 ######                       *##
            #,              ###########                       #
            #####################################          ####
            #####################################      &#######
            #####################################  .###########
            # print("1.clustersFinal\n",len(clustersFinal),clustersFinal)
                                        
            if len(dcSim)==0:
                for dc in range(len(dccil)):
                    sim = 0 
                    for cl in range(n):
                        if newGmadc.getSim(dc,cl)>sim: # and dccil[dc]!=cl:
                            sim = newGmadc.getSim(dccil[dc],cl)
                            if len(dcSim)>dc:
                                dcSim[dc]=cl
                            else:
                                dcSim.append(cl) 

            # print(dcSim)
            # print(clustersFinal)
            for dc in range(min(len(dcSim), len(dccil))):
                DCposition = index_2d(clustersFinal, dccil[dc])
                # print("dc: ",dc,"\nlen(dccil): ",len(dccil),"\nlen(dcSim): ",len(dcSim))
                DSposition = index_2d(clustersFinal, dcSim[dc])
                clustersFinal[DSposition[0]].append(clustersFinal[DCposition[0]].pop(DCposition[1]))
                print(" {}({})\t-->\t{}({})\t(Disconnected class: {},\tmoved to cluster of it's must similar class: {})" \
                    .format(DCposition[0],dccil[dc],
                            DSposition[0],dcSim[dc],
                            fileNames[dccil[dc]],
                            fileNames[dcSim[dc]]
                    ))
                    
            dccil=list()

            ###################################################



            time2 = time.time()
            timeClustering = time2 - time1

            timeTotal = timeInit + timeClustering


            # i disabled it
            # print(cvsFilePathString[cvsFileNumber]+ ":\tclustering completed at k = " + str(k)+ " and in run = " + str(run_no) + " :)")
            
            
            # exportingToMoJoFormatAlgorithmManual(k, n, clustersFinal)
            # print("exporting to MoJo format algorithm manually completed :)")
            
            exportingToMoJoFormatAlgorithm(k, n, clustersFinal, fileNames, filePathNames, run_no, pathResultString[cvsFileNumber])
            # print("exporting to MoJo format algorithm completed :)")

            MoJoAlgorithmPath ="{}/MoJoAlgorithm{}_{}.txt".format(pathResultString[cvsFileNumber],k,run_no)
            MoJoExpertPath= "{}/MoJoExpert.txt".format(pathResultString[cvsFileNumber])
            # print(run_no,k,cvsFileNumber,MoJoAlgorithmPath,MoJoExpertPath)
            proc = subprocess.Popen(["java", "mojo/MoJo", MoJoAlgorithmPath , MoJoExpertPath], stdout=subprocess.PIPE)
            outs, errs = proc.communicate()

            mojoMeasure =  int(outs[:-1])


            proc = subprocess.Popen(["java", "mojo/MoJo", MoJoAlgorithmPath ,  MoJoExpertPath,"-fm"], stdout=subprocess.PIPE)
            outs, errs = proc.communicate()

            mojoFmMeasure =  float(outs[:-1])

            # print(mojoFmMeasure)
            result.append([run_no,k,mojoMeasure,mojoFmMeasure,timeInit,timeClustering,timeTotal])
    # print(result)
    # bar.finish()
    outputFileResult = open (pathResultString[cvsFileNumber] + "/result.txt", "w")

    outputFileResult.write("RunNO\tK\tMoJo\tMoJo fm\tTime Init\tTime Clustering\tTime Total\n")
    # print("outputFileResult")
    mj,mjfm=list(),list()

    for i in range (0, maxRunNo * (maxK-1)):
        outputFileResult.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            result[i][0],
            result[i][1],
            result[i][2],
            result[i][3],
            result[i][4],
            result[i][5],
            result[i][6]
        ))
        mjfm.append(result[i][3])
        mj.append(result[i][2])
    outputFileResult.close()
    print("min(mj): ",min(mj))
    print("max(mjfm): ",max(mjfm))
    outputFileResult.close()

    del fileNames
    del filePathNames
    del cdgNonSymetric
    del SymetricCDG
    del similarityMatrix
    del dist_matrix
    del clustersFinal
    del clustersInit
    del clustersUpdateCenter
