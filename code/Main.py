# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:58 2018

@author: kyleq
"""
import pandas as pd
import numpy as np
import gc
from CarParkMap import CarParkMap
from Nearest import NearestAlgorithm
from Greedy1 import GreedyAlgorithm1

from Genetic1 import GeneticAlgorithm
from Genetic2 import GeneticAlgorithm2
from Cuckoo import CuckooAlgorithm
from PSO1 import PSOAlgorithm
from PSO2 import PSOAlgorithm2

from GA_a import GaArea
from GA_o import GaOfficer
from GA_oa import GaOfficerArea
from GA_oa_cluster import GaOfficerAreaCluster

# Records number per day
# 02_08 - 1795 (420, 1140)
# 02_09 - 2152 (420, 1140)
# 02_10 - 2143 (420, 1140)
# 02_11 - 2129 (420, 1140)
# 02_12 - 2128 (420, 1140)
# 02_13 - 1470 (420, 1140)
# 02_14 - 1764 (420, 1140)

# 02_08 - 1657 (480, 1080)
# 02_09 - 1970 (480, 1080)
# 02_10 - 1972 (480, 1080)
# 02_11 - 1976 (480, 1080)
# 02_12 - 1940 (480, 1080)
# 02_13 - 1279 (480, 1080)
# 02_14 - 1643 (480, 1080)

dataDate = '02_08'
# dayRange = '02_08-12'  # 02_08-12; 02_15-19;
mainPath = 'data/2016/'
dataSubset = 'full_nodes/'
resultPath = mainPath + dataSubset + 'results/' + dataDate + '/'
# nodesFileName = mainPath + dataSubset + 'bay_sensors_vio_loc_' + dayRange + '.csv'
nodesFileName = mainPath + dataSubset + 'bay_sensors_vio_loc_' + dataDate + '.csv'
distanceFileName = mainPath + dataSubset + 'dis_CBD_twoPs_' + dataDate + '.csv'
vioRecordsFileName = mainPath + dataSubset + 'bay_vio_data_' + dataDate + '.csv'
stayProFileName = mainPath + 'stayPro_jan.csv'

nodesData = pd.read_csv(nodesFileName)
disData = pd.read_csv(distanceFileName)
vioData = pd.read_csv(vioRecordsFileName)
stayProData = pd.read_csv(stayProFileName)

name_area=[]
num_Area=0
street_id=[]
line_num=0
for i in nodesData['area']:
    line_num+=1
    if i not in name_area:
        num_Area+=1
        name_area.append(i)
area_id=[j for j in range(num_Area)]
area_dic=dict(zip(name_area,area_id))
area_list=[0 for k in range(num_Area)]

timeRange = (480, 1080)  # 480 - 8am; 780 - 12:30pm; 800 - 1pm; 1080 - 6pm; 1140 - 7pm
vioData = vioData[(vioData.vioTime > timeRange[0]) & (vioData.vioTime < timeRange[1])]

# read distribution of leaving probability in the current area
leavingFileName = mainPath + 'leaving_pro.csv'
leaveProList = np.genfromtxt(leavingFileName, delimiter=',')
stayProList = [1 - x for x in leaveProList]
# stayProList = stayProData['pro'].tolist()

carParkMap = CarParkMap()
carParkMap.initialMap(nodesData, disData, vioData, resultPath)


# delete data from memory
del nodesData
del disData
del vioData
gc.collect()

startTime = timeRange[0] + 1  # Officer begin to collect fines from this time
endTime = timeRange[1]  # Officer end collecting fines at this time
updateTime = 1  # the violation information update frequency in minutes
speed = 70  # officers' avg speed
startPointMarker = 'central_station'  # (-37.810393, 144.964267, 'central_station')
algorithm = 12
officerNum = 30 # 15, 20, 30, 40, 50

if algorithm == 0:
    algorithm = NearestAlgorithm(startTime, endTime, updateTime, carParkMap,area_dic,area_list)
    algorithm.initialOfficers(speed, startPointMarker, officerNum)
    algorithm.execute()

if algorithm == 1:
    greedyAlgorithm = GreedyAlgorithm1(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    greedyAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    greedyAlgorithm.execute()


if algorithm == 3:
    geneticAlgorithm = GeneticAlgorithm(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    geneticAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    geneticAlgorithm.execute()

if algorithm == 4:
    geneticAlgorithm = GeneticAlgorithm2(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    geneticAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    geneticAlgorithm.execute()

if algorithm == 5:
    cuckooAlgorithm = CuckooAlgorithm(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    cuckooAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    cuckooAlgorithm.execute()


if algorithm == 7:
    psoAlgorithm = PSOAlgorithm(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    psoAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    psoAlgorithm.execute()

# if algorithm == 8:
#     psoAlgorithm2 = PSOAlgorithm2(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
#     psoAlgorithm2.initialOfficers(speed, startPointMarker, officerNum)
#     psoAlgorithm2.execute()


if algorithm==11:
    gaoffiAlgorithm=GaOfficer(startTime, endTime, updateTime, carParkMap, stayProList)
    gaoffiAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    gaoffiAlgorithm.execute()

if algorithm==12:
    gaareaAlgorithm = GaArea(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    gaareaAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    gaareaAlgorithm.execute()

if algorithm==13:
    gaOffiAreaAlgorithm = GaOfficerArea(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    gaOffiAreaAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    gaOffiAreaAlgorithm.execute()
