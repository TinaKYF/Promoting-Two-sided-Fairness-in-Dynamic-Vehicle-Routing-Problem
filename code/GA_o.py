from Officer import Officer
from random import uniform
from random import randint
from pandas import Series
import numpy as np
import Log as log
import copy
import gc
import datetime
import statistics
import time
from memory_profiler import profile

class GaOfficer:
    """
    LB Random key
    """


    def __init__(self, startTime=0, endTime=0, updateTime=1, carParkMap=None, stayProList=None,area_dic=None,area_list=None):
        self.startTime = startTime
        self.currentTime = startTime
        self.endTime = endTime
        self.updateTime = updateTime
        self.carParkMap = carParkMap
        self.stayProList = stayProList
        self.freeOfficers = None
        self.currentEdges = None
        self.currentNodes = None
        self.populations = []
        self.cells_id = []
        self.freeOfficersNum = 0
        self.solutionLen = 0
        self.populationSize = 100
        self.crossRate = 0.3
        self.mutateRate = 0.2
        self.elitistRate = 0.2
        self.pointNumLoc = 3
        self.doLocRate = 0.5
        self.maxGen = 300
        self.proThreshold = 0.21
        self.validCapturedTime = 600  # maximum is 600
        self.totalBenefit = 0
        self.seed=1234
        self.area_dic = area_dic
        self.area_list = area_list
        self.area_cap_list = copy.deepcopy(area_list)

    def initialOfficers(self, speed, startPointMarker, officerNum):
        for i in range(officerNum):
            self.carParkMap.officers.append(Officer(i, speed, startPointMarker))

    def execute(self):
        runTime = datetime.datetime.now()
        while self.currentTime < self.endTime:
            log.debugTag('currentTime', self.currentTime)
            self.updateOfficersStatus()
            self.assignNextPToOfficers()
            self.currentTime += self.updateTime
        stopTime = datetime.datetime.now()

        officerdis_record = []
        for officer in self.carParkMap.officers:
            officerdis_record.append(officer.totalDis)
        stdofficerdis_record = statistics.stdev(officerdis_record)

        areacaprate = []
        for a, b in zip(self.area_list, self.area_cap_list):
            if a == 0:
                a = 1
            caprate = b / a
            areacaprate.append(caprate)
        stdareacaprate = statistics.stdev(areacaprate)
        # write results to file
        fileName = 'Fair_officer_result'
        titles = ['runTime', 'stopTime', 'startTime', 'endTime', 'updateTime', 'popuSize', 'crossRate', 'mutateRate',
                  'elitistRate', 'pointNumLoc', 'doLocRate', 'maxGen', 'proThreshold', 'validCapturedTime','officerdis_record','stdofficerdis_record','area_vionum','area_capnum','area_caprate','stdarea_caprate']
        params = [runTime, stopTime, self.startTime, self.endTime, self.updateTime, self.populationSize, self.crossRate,
                  self.mutateRate, self.elitistRate, self.pointNumLoc, self.doLocRate, self.maxGen, self.proThreshold,
                  self.validCapturedTime,officerdis_record,stdofficerdis_record,self.area_list,self.area_cap_list,areacaprate,stdareacaprate]
        self.carParkMap.printAllResults(fileName, titles, params)

    def updateOfficersStatus(self):
        for officer in self.carParkMap.officers:
            # when an officer finish his traveling
            if officer.assigned is True and officer.arriveTime <= self.currentTime:
                officer.assigned = False
                self.carParkMap.releaseNode(officer.occupiedMarker)

    def assignNextPToOfficers(self):
        # initialise the population(randomly)
        self.initialPopulations()

        # assign and get rewards
        if self.populations:
            self.evolvePopulation()

            pathBest = self.populations[0][0]
            pathLen = len(pathBest)

            for i, ele in enumerate(pathBest):
                # if it is an officer Id, and the following ids are nodes that are assigned to it
                if 'O' in ele[0]:
                    officer = self.carParkMap.findOfficerById(ele[0])
                    hasNextNode = False
                    nextPMarker = ''
                    if i + 1 < pathLen:
                        nextEle = pathBest[i + 1]
                        if 'O' not in nextEle[0]:
                            hasNextNode = True
                            nextPMarker = nextEle[0]

                    if hasNextNode is True:
                        # update next position for officer
                        dis = self.getDistance(officer.occupiedMarker, nextPMarker)
                        travelTime = dis / officer.walkingSpeed
                        officer.arriveTime = self.currentTime + travelTime
                        officer.myPath.append(nextPMarker)
                        officer.occupiedMarker = nextPMarker
                        officer.assigned = True
                        officer.totalDis += dis

                        # update node's record status
                        node = self.carParkMap.findNodeByMarkerId(nextPMarker)
                        node.assigned = True
                        depTime = node.removeRecordByTime(self.currentTime)

                        # benefit increase when officer arrive before leaving for this record
                        if officer.arriveTime < depTime:
                            officer.benefit += 1
                            officer.validDis += dis
                            self.area_cap_list[self.area_dic[node.area]] += 1
                            self.totalBenefit += 1
                    else:
                        officer.saveTime += self.updateTime
        else:
            for officer in self.freeOfficers:
                officer.saveTime += self.updateTime
        # # record benefits per minute
        # results = [[self.currentTime, self.totalBenefit]]
        # with open('25_offi_genetic2_solution_result.csv', "a") as output:
        #     writer = csv.writer(output, lineterminator='\n')
        #     writer.writerows(results)

    def initialPopulations(self):
        self.freeOfficers = self.carParkMap.getFreeOfficers()
        self.currentNodes = self.carParkMap.getFreeVioNodes(self.currentTime)
        log.debugTag('freeOfficers', len(self.freeOfficers))
        log.debugTag('currentNodes', len(self.currentNodes))
        self.populations.clear()
        self.currentcountNodes = self.carParkMap.getnotcountVioNodes(self.currentTime)
        if self.currentcountNodes:
            for node in self.currentcountNodes:
                self.area_list[self.area_dic[node.area]] += 1
                node.removecountRecord(self.currentTime)

        if self.freeOfficers and self.currentNodes:
            nodesId = [n.marker for n in self.currentNodes]
            self.cells_id = [o.id_ for o in self.freeOfficers] + nodesId
            self.solutionLen = len(self.cells_id)
            self.freeOfficersNum = len(self.freeOfficers)
            self.getCurrentEdges(nodesId)
            offisLeader = 0

            if len(self.freeOfficers) == 1:
                size = 1
            else:
                size = self.populationSize

            for i in range(size):
                if offisLeader == self.freeOfficersNum:
                    offisLeader = 0
                newSolution = self.generateNewSolution(offisLeader)
                offisLeader += 1
                self.populations.append(copy.deepcopy(newSolution))
            self.populations.sort(key=lambda tup: -tup[1])
            # delete variables from memory
            gc.collect()


    def evolvePopulation(self):
        if self.freeOfficersNum == 1:
            self.localOptimisation(self.populations[0][0])  # local optimisation
        else:
            for t in range(self.maxGen):
                log.debugTag('evolvePopulation2', t)
                newPopulations = copy.deepcopy(self.populations)
                elitistNum = self.populationSize * self.elitistRate
                crossNum = self.populationSize * self.crossRate
                mutateNum = self.populationSize * self.mutateRate

                populationscopy = copy.deepcopy(self.populations)
                time1=time.time()
                self.localstddis(populationscopy)
                time2=time.time()
                self.computerStddis(populationscopy)
                time3=time.time()
                populationscopy.sort(key=lambda tup: tup[2])
                print(time2-time1)
                print(time3-time2)
                # print('stddis generated')
                # Crossover
                for i in range(int(elitistNum), int(elitistNum + crossNum)):
                    newChild = self.generateChildChro(populationscopy)
                    if uniform(0, 1) > self.doLocRate:
                        self.localOptimisation(newChild)
                    pro = self.computeAveragePro(newChild)
                    newPopulations[i] = [copy.deepcopy(newChild), pro]

                # Immigration - Like typical mutation
                offisLeader = 0
                mutateStart = self.populationSize - int(mutateNum)
                for j in range(mutateStart, self.populationSize):
                    if offisLeader == self.freeOfficersNum:
                        offisLeader = 0
                    newSolution = self.generateNewSolution(offisLeader)
                    offisLeader += 1
                    newPopulations[j] = copy.deepcopy(newSolution)

                self.populations.clear()
                self.populations = newPopulations
                self.populations.sort(key=lambda tup: -tup[1])

    def generateNewSolution(self, offisLeader):
        # np.random.seed(self.seed)
        agents = np.random.uniform(0, 1, size=self.solutionLen).tolist()
        agents[offisLeader] = -5000  # decide an officer to be a leade·r in a solution
        initPath = list(map(lambda x, y: [x, y], self.cells_id, agents))
        initPath.sort(key=lambda tup: tup[1])  # sort path by random number
        # if uniform(0, 1) > self.doLocRate:
        #     self.localOptimisation(initPath)
        pro = self.computeAveragePro(initPath)
        return [initPath, pro]

    def generateChildChro(self,populations):

        crossNum = self.populationSize * self.crossRate
        child1Chro = []
        a = -1
        b = -1
        # Select parents, a and b must be different
        while a == b:
            a = randint(0, crossNum - 1)
            b = randint(0, crossNum - 1)
        parent1Chro = copy.deepcopy(populations[a][0])
        parent2Chro = copy.deepcopy(populations[b][0])

        # get a leader element first
        leaderId = parent1Chro[0][0]
        child1Chro.append(parent1Chro[0])

        parent1Chro.sort(key=lambda tup: tup[0])  # sort path by id
        parent2Chro.sort(key=lambda tup: tup[0])  # sort path by id

        # log.debugTag('solutionLen', self.solutionLen)
        for i in range(self.solutionLen):
            if parent1Chro[i][0] == leaderId:
                continue
            if uniform(0, 1) <= self.crossRate:
                child1Chro.append(parent1Chro[i])
            else:
                child1Chro.append(parent2Chro[i])
        child1Chro.sort(key=lambda tup: tup[1])  # sort path by random number

        # set only first element to be leader element
        if child1Chro[1][1] == -5000:
            child1Chro[1][1] = uniform(0, 1)
            child1Chro.sort(key=lambda tup: tup[1])  # sort path by random number
        return child1Chro

    def getCurrentEdges(self, nodesId):
        twoPs = []
        append = twoPs.append
        for o in self.freeOfficers:  # officers to vio nodes
            n1 = o.occupiedMarker
            for n2 in nodesId:
                append(n1 + '_' + n2)
                append(n2 + '_' + n1)

        length = len(nodesId)
        for i in range(length):  # nodes to vio nodes
            k = i + 1
            while k < length:
                append(nodesId[i] + '_' + nodesId[k])
                append(nodesId[k] + '_' + nodesId[i])
                k += 1
        edges = self.carParkMap.findEdgeByNodes(twoPs)
        self.currentEdges = Series(edges.distance.values, index=edges.twoPs).to_dict()
        # delete variables from memory
        del twoPs
        del edges

    def getDistance(self, startPMarker, toPMarker):
        if startPMarker == toPMarker:
            return 0
        # log.debugTest(startPMarker)
        key = startPMarker + '_' + toPMarker
        if key in self.currentEdges:
            return self.currentEdges[key]
        else:
            key = toPMarker + '_' + startPMarker
            return self.currentEdges[key]

    def getCurrentNodeById(self, markerId):
        for node in self.currentNodes:
            if node.marker == markerId:
                return node
        return None

    def findFreeOfficer(self, id_):
        for offi in self.freeOfficers:
            if offi.id_ == id_:
                return offi
        return None

    def computeAveragePro(self, path):
        propTotal = 0.0
        assumedCapturedNum = 0
        assumedCurTime = 0
        officer = None
        currentNodeId = None
        pointsCount = 0

        for ele in path:
            # O indicate that it is an officer Id, and the following ids are nodes that will be assigned
            if 'O' in ele[0]:
                officer = self.findFreeOfficer(ele[0])
                currentNodeId = officer.occupiedMarker
                assumedCurTime = self.currentTime
                pointsCount = 0
            else:
                pointsCount += 1
                if pointsCount >= self.pointNumLoc:
                    continue
                if assumedCurTime - self.currentTime > self.validCapturedTime:
                    continue
                nextNode = self.getCurrentNodeById(ele[0])
                travelTime = self.getDistance(currentNodeId, nextNode.marker) / officer.walkingSpeed
                arriveTime = assumedCurTime + travelTime
                probability, recordId = nextNode.calculateProbability(arriveTime, self.currentTime, self.stayProList)
                assumedCurTime = arriveTime
                currentNodeId = nextNode.marker
                if probability > self.proThreshold:
                    propTotal += probability
                    assumedCapturedNum += 1
        if assumedCapturedNum == 0:
            return 0
        else:
            return propTotal / assumedCapturedNum

    def localOptimisation(self, path):
        """
        Reorder points in each sub path with max pro from the previous node
        :param path: a path reference
        """
        offiIndex = 0
        pathEndNodeI = len(path) - 1

        for i, ele in enumerate(path):  # ele will be ['3357s', 1.2345]
            if 'O' in ele[0] or i == pathEndNodeI:
                if 'O' in ele[0]:  # if it is an officer Id
                    subPathEndI = i - 1
                else:  # if it is a last node Id
                    subPathEndI = i

                nodesNum = subPathEndI - offiIndex  # number of nodes in subpath
                if nodesNum > 1:
                    currentNodeId = self.findFreeOfficer(path[offiIndex][0]).occupiedMarker
                    assumedCurTime = self.currentTime
                    # concern on next 3 points only
                    if nodesNum > self.pointNumLoc:
                        nodesNum = self.pointNumLoc

                    for j in range(1, nodesNum):  # compare n - 1 times
                        if assumedCurTime - self.currentTime > self.validCapturedTime:
                            break
                        proMax = -2
                        winner = None
                        nStartI = offiIndex + j

                        for k in range(nStartI, subPathEndI + 1):  # indexes on the sub path
                            pId = path[k][0]
                            travelTime = self.getDistance(currentNodeId, pId) / 70
                            arriveTime = assumedCurTime + travelTime
                            nextNode = self.getCurrentNodeById(pId)
                            pro, recordId = nextNode.calculateProbability(arriveTime, self.currentTime,
                                                                          self.stayProList)
                            if pro > proMax:
                                proMax = pro
                                winner = [k, pId, arriveTime]
                        winnerIndex = winner[0]
                        currentNodeId = winner[1]  # update current node with winner node
                        assumedCurTime = winner[2]  # update current time
                        if nStartI != winnerIndex:
                            path[nStartI][0], path[winnerIndex][0] = path[winnerIndex][0], path[nStartI][0]
                offiIndex = i  # start from cur office if the former offi has no or one node

    # def localstddis(self, population):
    #     """
    #     Reorder points in each sub path with max pro from the previous node
    #     :param path: a path reference
    #     """
    #     officerdis=[]
    #     for officer in self.carParkMap.officers:
    #         officerdis.append(officer.totalDis)
    #
    #     for popuindex in range(len(population)):
    #         solution=population[popuindex]
    #         path=solution[0]
    #         offiIndex = 0
    #         pathEndNodeI = len(path) - 1
    #
    #         for i, ele in enumerate(path):  # ele will be ['3357s', 1.2345]
    #             if 'O' in ele[0] or i == pathEndNodeI:
    #                 if 'O' in ele[0]:  # if it is an officer Id
    #                     subPathEndI = i - 1
    #                 else:  # if it is a last node Id
    #                     subPathEndI = i
    #
    #                 nodesNum = subPathEndI - offiIndex  # number of nodes in subpath
    #                 if nodesNum > 1:
    #                     currentNodeId = self.findFreeOfficer(path[offiIndex][0]).occupiedMarker
    #                     for j in range(1,nodesNum):
    #                         stdmax=float("inf")
    #                         winner=None
    #                         nStartI=offiIndex+j
    #                         for k in range(nStartI,subPathEndI+1):
    #                             pId=path[k][0]
    #                             dis=self.getDistance(currentNodeId,pId)
    #                             nextNode = self.getCurrentNodeById(pId)
    #                             officerdiscopy=copy.deepcopy(officerdis)
    #                             officerid=path[offiIndex][0]
    #                             officeridnum=int(officerid[1:])
    #                             officerdiscopy[officeridnum]+=dis
    #                             std=statistics.stdev(officerdiscopy)
    #                             if std<stdmax:
    #                                 stdmax=std
    #                                 winner=[k,pId]
    #                         winnerIndex=winner[0]
    #                         currentNodeId=winner[1]
    #                         if nStartI!=winnerIndex:
    #                             path[nStartI][0], path[winnerIndex][0] = path[winnerIndex][0], path[nStartI][0]
    #                 offiIndex=i
        # print('localstddis')
    def computerStddis(self,population):
        officerdis = []
        for officer in self.carParkMap.officers:
            officerdis.append(officer.totalDis)
        for popuindex in range(len(population)):
            solution=population[popuindex]
            path=solution[0]
            officerdiscopy=copy.deepcopy(officerdis)
            for eleindex in range(len(path)-1):
                # O indicate that it is an officer Id, and the following ids are nodes that will be assigned
                if 'O' in path[eleindex][0]:
                    officer = self.findFreeOfficer(path[eleindex][0])
                    if 'O' not in path[eleindex+1][0]:
                        nodeid=path[eleindex+1][0]
                        dis=self.getDistance(officer.occupiedMarker,nodeid)
                        offiecrid=int(path[eleindex][0][1:])
                        officerdiscopy[offiecrid]+=dis
            std=statistics.stdev(officerdiscopy)
            solution.append(std)
        # print('computerStddis')
  # start from cur office if the former offi has no or one node

    def localstddis(self, population):
        """
        Reorder points in each sub path with max pro from the previous node
        :param path: a path reference
        """
        officerdis=[]
        for officer in self.carParkMap.officers:
            officerdis.append(officer.totalDis)

        for popuindex in range(len(population)):
            solution=population[popuindex]
            path=solution[0]
            offiIndex = 0
            pathEndNodeI = len(path) - 1

            for i, ele in enumerate(path):  # ele will be ['3357s', 1.2345]
                if 'O' in ele[0] or i == pathEndNodeI:
                    if 'O' in ele[0]:  # if it is an officer Id
                        subPathEndI = i - 1
                    else:  # if it is a last node Id
                        subPathEndI = i

                    nodesNum = subPathEndI - offiIndex  # number of nodes in subpath
                    if nodesNum > 1:
                        currentNodeId = self.findFreeOfficer(path[offiIndex][0]).occupiedMarker
                        stdmax = float("inf")
                        winner = None
                        nStartI=offiIndex+1
                        for j in range(nStartI,subPathEndI + 1):

                            # nStartI=offiIndex+j
                            # for k in range(nStartI,subPathEndI+1):
                            pId=path[j][0]
                            dis=self.getDistance(currentNodeId,pId)
                            nextNode = self.getCurrentNodeById(pId)
                            officerdiscopy=copy.deepcopy(officerdis)
                            officerid=path[offiIndex][0]
                            officeridnum=int(officerid[1:])
                            officerdiscopy[officeridnum]+=dis
                            std=statistics.stdev(officerdiscopy)
                            if std<stdmax:
                                stdmax=std
                                winner=j
                        if nStartI!=winner:
                            path[nStartI][0], path[winner][0] = path[winner][0], path[nStartI][0]
                    offiIndex=i