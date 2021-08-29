#=============================================================================#
#    GMADCC Algorythm                                                         #
#    To solve random choise on disconected Softwere class graps.              #
#    By: Masoud Azizi     Email: mablue92@gmail.com                           #
#=============================================================================#

import builtins
import inspect
import ntpath
import os
import random
import re
import string
from difflib import SequenceMatcher
from time import time


class GMADisconnectedClassClassifier:
    
    paths=None
    disconectedClassesIndex=None

    #=============================================================================#
    # Vareables

    def __init__(self,paths):
        self.paths=paths
        super().__init__()

    #=============================================================================#

    def longest_common_substring(self, s1, s2):
        m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
        longest, x_longest = 0, 0
        for x in range(1, 1 + len(s1)):
            for y in range(1, 1 + len(s2)):
                if s1[x - 1] == s2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                        x_longest = x
                else:
                    m[x][y] = 0
        return s1[x_longest - longest: x_longest]

    # Get longest common substring as precents
    def similarity(self,s1, s2):
        # longest_common_substring * 2 cuz we have this string in all two strings(s1 and s2)
        return (len(self.longest_common_substring(s1, s2))*2) / (len(s1) + len(s2)) * 100

    #=============================================================================#

    def diff(self, str1, str2):
        return SequenceMatcher(lambda x: x == " ", str1 or "", str2 or "").ratio()*100

    #=============================================================================#

    def setMaxMatchFinder(self, set1, set2):
        lenSet1 = len(set1)
        lenSet2 = len(set2)
        smallestSet = min(lenSet1, lenSet2)
        setSimilarity = len(set(set1) & set(set2))
        return 100*setSimilarity/smallestSet

    #=============================================================================#

    def getFileContants(self, path):
        # path=os.path.join("SourceCodes/gfx.src/",path)
        f = open(path.replace("\\\\","/").replace("\\","/").replace("+",""), "r", errors='ignore')
        strInc = "#include "
        
        # filecontant = f.read()
        # 
        Lines = f.readlines()
        filecontant=list()
        # Strips the newline character
        for line in Lines:
            if line.startswith(strInc):
                filecontant.append(line.strip().replace("\"","")[len(strInc):])
                # print(filecontant)

        f.close()
        return filecontant

    def getFileName(self,path):
        return ntpath.basename(path)
        
    #=============================================================================#

    def getSim(self,ci,oci):
        className = self.getFileName(self.paths[ci])
        classText = self.getFileContants(self.paths[ci])
        otherClassName = self.getFileName(self.paths[oci])
        otherClassText = self.getFileContants(self.paths[oci])
        sim = round(
                        self.setMaxMatchFinder(classText, otherClassText)/4+  # good
                        self.diff(classText, otherClassText)/4+  # Algo runtime takes 22.560689 secends, so I ebabled it \
                        self.similarity(className, otherClassName)/8+  # good # Algo runtime takes 0.208008 secends, so I ebabled it
                        # similarity(classText, otherClassText)/10+  #+ very massive algo, so I disabled it
                        self.diff(className, otherClassName)/8  # goood # Algo runtime takes 0.379022 secends, so I disabled it
                        , 2)
        return sim
