import pandas as pd
import numpy as np
import gc
from CarParkMap import CarParkMap
# from Nearest import NearestAlgorithm
# from Greedy1 import GreedyAlgorithm1
# from Genetic1 import GeneticAlgorithm
# from Optimal import OptimalAlgorithm
from GA_oa_cluster import GaOfficerAreaCluster
from memory_profiler import profile
from cluster import CLUSTER
from GA_a_cluster import GaClusterArea
from GA_o_cluster import GaClusterOfficer
from Genetic11 import GeneticAlgorithm11


dataDate = '02_08'
mainPath = 'data/2016/'
dataSubset = 'full_nodes/'
resultPath = mainPath + dataSubset + 'results/' + dataDate + '/'
nodesFileName = mainPath + dataSubset + 'bay_sensors_vio_loc_' + dataDate + '.csv'
distanceFileName = mainPath + dataSubset + 'dis_CBD_twoPs_' + dataDate + '.csv'
vioRecordsFileName = mainPath + dataSubset + 'bay_vio_data_' + dataDate + '.csv'
stayProFileName = mainPath + 'stayPro_jan.csv'

nodesData = pd.read_csv(nodesFileName)
disData = pd.read_csv(distanceFileName)
vioData = pd.read_csv(vioRecordsFileName)
stayProData = pd.read_csv(stayProFileName)

name_area = []
num_Area = 0
street_id = []
line_num = 0
for i in nodesData['area']:
    line_num += 1
    if i not in name_area:
        num_Area += 1
        name_area.append(i)
area_id = [j for j in range(num_Area)]
area_dic = dict(zip(name_area, area_id))
area_list = [0 for k in range(num_Area)]

officerNum = 20 # 15, 20, 30, 40, 50
Clusterhisdata=CLUSTER(officerNum)
# print(Clusterhisdata)
startPointMarker=Clusterhisdata.cluster_hisdata()
print(startPointMarker)

timeRange = (480, 1080)  # 480 - 8am; 1080 - 6pm
# timeRange = (800, 1140)  # 480 - 1pm; 1140 - 7pm
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
# startPointMarker = 'central_station'  # (-37.810393, 144.964267, 'central_station')


# for 50 officers
algorithm = 12

if algorithm == 6:
    LerkGenetic1 = GaOfficerAreaCluster(startTime, endTime, updateTime, carParkMap, stayProList, area_dic, area_list)
    LerkGenetic1.initialOfficers(speed, startPointMarker, officerNum)
    LerkGenetic1.execute()

if algorithm==11:
    gaoffiAlgorithm=GaClusterOfficer(startTime, endTime, updateTime, carParkMap, stayProList,area_dic, area_list)
    gaoffiAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    gaoffiAlgorithm.execute()

if algorithm==12:
    gaareaAlgorithm = GaClusterArea(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    gaareaAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    gaareaAlgorithm.execute()

if algorithm==21:
    geneticAlgorithm=GeneticAlgorithm11(startTime, endTime, updateTime, carParkMap, stayProList,area_dic,area_list)
    geneticAlgorithm.initialOfficers(speed, startPointMarker, officerNum)
    geneticAlgorithm.execute()