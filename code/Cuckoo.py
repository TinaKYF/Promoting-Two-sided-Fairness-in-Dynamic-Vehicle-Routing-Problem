# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:58 2018

@author: kyleq
"""
from Officer import Officer
from random import randint
from random import uniform
from pandas import Series
import numpy as np
import math
import copy
import Log as log
import gc
import datetime
import statistics
import copy


class CuckooAlgorithm:
    """
    In initialisation, each officer will be a leader on one population (solution) once. And one solution will
    contain both current free officer and vio nodes with random number to indicate the order position. After that,
    the populations will be reordered by doubleBridgeMove or twoOptMove through Levy Flight.
    """

    def __init__(self, startTime=0, endTime=0, updateTime=1, carParkMap=None, stayProList=None,area_dic=None,
                 area_list=None):
        self.startTime = startTime
        self.currentTime = startTime
        self.endTime = endTime
        self.updateTime = updateTime
        self.carParkMap = carParkMap
        self.stayProList = stayProList
        self.currentEdges = None
        self.currentNodes = None
        self.freeOfficers = None
        self.nests = []
        self.nestSize = 100  # bigger is better
        self.pc = 0.6
        self.pa = 0.3  # bigger is better
        self.preFlyRate = 0.3  # 0.3 is better for larger dataset
        self.pointNumLoc = 3
        self.doLocRate = 0.5
        self.maxGen = 300
        self.lambda_ = 1
        self.stepSide = 0.05
        self.proThreshold = 0.21  # not bigger than 0.5
        self.totalBenefit = 0
        self.validCapturedTime = 600  # maximum is 600
        self.area_dic = area_dic
        self.area_list = area_list
        self.area_cap_list = copy.deepcopy(area_list)

    def initialOfficers(self, speed, startPointMarker, officerNum):
        for i in range(officerNum):
            self.carParkMap.officers.append(Officer(i, speed, startPointMarker))

    def execute(self):
        runTime = datetime.datetime.now()
        while self.currentTime < self.endTime:
            log.infoTag('currentTime', self.currentTime)
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
        fileName = 'cuckoo_result.csv'
        titles = ['runTime', 'stopTime', 'startTime', 'endTime', 'updateTime', 'nestSize', 'pc', 'pa', 'preFlyRate',
                  'pointNumLoc', 'doLocRate', 'lambda_', 'stepSide', 'maxGen', 'proThreshold', 'validCapturedTime','officerdis_record', 'stdofficerdis_record', 'area_vionum', 'area_capnum', 'area_caprate',
                  'stdarea_caprate']
        params = [runTime, stopTime, self.startTime, self.endTime, self.updateTime, self.nestSize,
                  self.pc, self.pa, self.preFlyRate, self.pointNumLoc, self.doLocRate, self.lambda_, self.stepSide,
                  self.maxGen, self.proThreshold, self.validCapturedTime,officerdis_record, stdofficerdis_record,self.area_list, self.area_cap_list,
                  areacaprate, stdareacaprate]
        self.carParkMap.printAllResults(fileName, titles, params)

    def updateOfficersStatus(self):
        for officer in self.carParkMap.officers:
            # when an officer finish his traveling
            if officer.assigned is True and officer.arriveTime <= self.currentTime:
                officer.assigned = False
                self.carParkMap.releaseNode(officer.occupiedMarker)

    def assignNextPToOfficers(self):
        self.currentcountNodes = self.carParkMap.getnotcountVioNodes(self.currentTime)
        if self.currentcountNodes:
            for node in self.currentcountNodes:
                self.area_list[self.area_dic[node.area]] += 1
                node.removecountRecord(self.currentTime)
        # initialise the solutions (randomly)
        self.initialSolutions()

        # assign and get rewards
        if self.nests:
            self.searchBestNests()

            pathBest = self.nests[0][0]
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
        # with open('25_offi_cuckoo_solution_result.csv', "a") as output:
        #     writer = csv.writer(output, lineterminator='\n')
        #     writer.writerows(results)

    def initialSolutions(self):
        log.infoTest('initialSolutions')
        self.freeOfficers = self.carParkMap.getFreeOfficers()
        self.currentNodes = self.carParkMap.getFreeVioNodes(self.currentTime)
        log.debugTag('freeOfficers', len(self.freeOfficers))
        log.debugTag('vioNodes', len(self.currentNodes))
        self.nests.clear()

        if self.freeOfficers and self.currentNodes:
            nodesId = [n.marker for n in self.currentNodes]
            cellIds = [o.id_ for o in self.freeOfficers] + nodesId
            solutionLen = len(cellIds)
            offisLen = len(self.freeOfficers)
            offisLeader = 0
            self.getCurrentEdges(nodesId)

            if len(self.freeOfficers) == 1:
                size = 1
            else:
                size = self.nestSize

            for i in range(size):
                agents = np.random.uniform(0, 1, size=solutionLen).tolist()
                if offisLeader == offisLen:
                    offisLeader = 0
                agents[offisLeader] = -5000  # decide an officer to be a leader in a solution
                offisLeader += 1
                initPath = list(map(lambda x, y: [x, y], cellIds, agents))
                initPath.sort(key=lambda tup: tup[1])  # sort path by random number
                pro = self.computeAveragePro(initPath)
                self.nests.append([copy.deepcopy(initPath), pro])
            self.nests.sort(key=lambda tup: -tup[1])
            # delete variables from memory
            del cellIds
            gc.collect()

    def searchBestNests(self):
        if len(self.freeOfficers) == 1:
            self.localOptimisation(self.nests[0][0])  # local optimisation
        else:
            for t in range(self.maxGen):
                log.infoTag('searchBestNests ', str(t))
                betterNestSize = int(self.pc * self.nestSize)
                preFlySize = int(self.preFlyRate * self.nestSize)

                for i in range(preFlySize):
                    nestIndex = randint(0, betterNestSize)
                    path = copy.deepcopy(self.nests[nestIndex][0])
                    self.generateNewPath(path)  # path is a reference which has objects
                    if uniform(0, 1) > self.doLocRate:
                        self.localOptimisation(path)
                    pro = self.computeAveragePro(path)
                    replacedNestIndex = randint(0, self.nestSize - 1)
                    if self.nests[replacedNestIndex][1] < pro:
                        self.nests[replacedNestIndex] = [path, pro]
                        self.nests.sort(key=lambda tup: -tup[1])

                # log.getTime()
                # abandoned and new ones are built based on the best one
                paNum = int(self.pa * self.nestSize)
                for i in range(self.nestSize - paNum, self.nestSize):
                    nestIndex = randint(0, betterNestSize)
                    pathBest = self.nests[nestIndex][0]
                    newPath = copy.deepcopy(pathBest)
                    self.generateNewPath(newPath)  # path is a reference which has objects
                    # if uniform(0, 1) > self.doLocRate:
                    #     self.localOptimisation(newPath)
                    pro = self.computeAveragePro(newPath)
                    self.nests[i] = [newPath, pro]
                # log.timeDiffNow('built and abandoned')
                self.nests.sort(key=lambda tup: -tup[1])

    def generateNewPath(self, path):
        updateNodesNum = randint(1, len(path) - 1)  # Select randomly number of nodes to be updated
        for i in range(1, updateNodesNum):
            levyDis = self.stepSide * self.levyFlight()  # replace agent for the node on path
            while levyDis < -5000:
                levyDis = self.stepSide * self.levyFlight()
            path[i][1] += levyDis
        path.sort(key=lambda tup: tup[1])  # sort path by random number

    def assignNewLeader(self, path):
        maxPro = -1
        startIndex = 0
        assignNum = 1
        hasNewLeader = False

        for i, ele in enumerate(path):
            # O indicate that it is an officer Id, and the following ids are nodes that will be assigned to this
            # officer until another officer Id appeared in the path
            if 'O' in ele[0]:
                hasNewLeader = False
                officer = self.findFreeOfficer(ele[0])
                currentNodeId = officer.occupiedMarker
                nextIndex = i + 1

                if nextIndex < len(path) and 'O' not in path[nextIndex][0]:
                    nextNode = self.getCurrentNodeById(path[nextIndex][0])
                    travelTime = self.getDistance(currentNodeId, nextNode.marker) / officer.walkingSpeed
                    arriveTime = self.currentTime + travelTime
                    pro, recordId = nextNode.calculateProbability(arriveTime, self.currentTime, self.stayProList)
                    if pro > maxPro:
                        startIndex = i
                        maxPro = pro
                        hasNewLeader = True
                        assignNum = 1  # recalculate nodes when has a new leader
            elif hasNewLeader:
                assignNum += 1
        # move a given sub path to the front
        for j in range(assignNum):
            path.insert(0 + j, path.pop(startIndex + j))

        path[0][1] = -5000
        updateNodesNum = randint(1, len(path) - 1)  # Select randomly number of nodes to be updated
        for i in range(1, updateNodesNum):
            levyDis = self.stepSide * self.levyFlight()  # replace agent for the node on path
            while levyDis < -5000:
                levyDis = self.stepSide * self.levyFlight()
            path[i][1] += levyDis
        path.sort(key=lambda tup: tup[1])  # sort path by random number

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
            # O indicate that it is an officer Id, and the following ids of nodes are assigned to the
            # officer until another officer Id appeared in the path
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

    def levyFlight(self):
        """
        Generate step from levy distribution
        :return:
        """
        sigma1 = np.power((math.gamma(1 + self.lambda_) * np.sin((np.pi * self.lambda_) / 2)) /
                          math.gamma((1 + self.lambda_) / 2) * np.power(2, (self.lambda_ - 1) / 2), 1 / self.lambda_)
        sigma2 = 1
        u = np.random.normal(0, sigma1, 1)
        v = np.random.normal(0, sigma2, 1)
        step = u / np.power(np.fabs(v), 1 / self.lambda_)
        return step[0]
