# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:58 2018

@author: kyleq
"""
from Officer import Officer
from random import uniform
from operator import attrgetter
from pandas import Series
import numpy as np
import Log as log
import copy
import gc
import datetime
import statistics
import numpy


class PSOAlgorithm:
    """
    LB Random key
    """

    def __init__(self, startTime=0, endTime=0, updateTime=1, carParkMap=None, stayProList=None,area_dic=None,
                 area_list=None):
        self.startTime = startTime
        self.currentTime = startTime
        self.endTime = endTime
        self.updateTime = updateTime
        self.carParkMap = carParkMap
        self.stayProList = stayProList
        self.freeOfficers = None
        self.currentEdges = None
        self.currentNodes = None
        self.gBestParticle = None
        self.swarm = []
        self.cellIds = []
        self.freeOfficersLen = 0
        self.solutionLen = 0
        self.popuSize = 100
        self.minX = 0
        self.maxX = 1
        self.maxGen = 300
        self.w = 1.1  # inertia
        self.c1 = 0.1  # cognitive (particle)
        self.c2 = 0.2  # social (swarm)
        self.updateRate = 0.5
        self.pointNumLoc = 3
        self.doLocRate = 0.5
        self.proThreshold = 0.21
        self.validCapturedTime = 600  # maximum is 600
        self.totalBenefit = 0
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
        fileName = 'PSO_result.csv'
        titles = ['runTime', 'stopTime', 'startTime', 'endTime', 'updateTime', 'popuSize', 'w', 'c1', 'c2', 'minX',
                  'maxX', 'updateRate', 'pointNumLoc', 'doLocRate', 'maxGen', 'proThreshold', 'validCapturedTime','officerdis_record', 'stdofficerdis_record', 'area_vionum', 'area_capnum', 'area_caprate',
                  'stdarea_caprate']
        params = [runTime, stopTime, self.startTime, self.endTime, self.updateTime, self.popuSize, self.w, self.c1,
                  self.c2, self.minX, self.maxX, self.updateRate, self.pointNumLoc, self.doLocRate, self.maxGen,
                  self.proThreshold, self.validCapturedTime,officerdis_record, stdofficerdis_record,self.area_list, self.area_cap_list,
                  areacaprate, stdareacaprate]
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
        if self.swarm:
            self.evolveParticles()

            pathBest = self.gBestParticle.pBestSolution
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
                            self.totalBenefit += 1
                            self.area_cap_list[self.area_dic[node.area]] += 1
                    else:
                        officer.saveTime += self.updateTime
            # results = [[self.currentTime, self.totalBenefit]]
            # with open('70_offi_2_uptime_PSO_solution_result.csv', "a") as output:
            #     writer = csv.writer(output, lineterminator='\n')
            #     writer.writerows(results)
        else:
            for officer in self.freeOfficers:
                officer.saveTime += self.updateTime

    def initialPopulations(self):
        self.currentcountNodes = self.carParkMap.getnotcountVioNodes(self.currentTime)
        if self.currentcountNodes:
            for node in self.currentcountNodes:
                self.area_list[self.area_dic[node.area]] += 1
                node.removecountRecord(self.currentTime)
        self.freeOfficers = self.carParkMap.getFreeOfficers()
        self.currentNodes = self.carParkMap.getFreeVioNodes(self.currentTime)
        log.debugTag('freeOfficers', len(self.freeOfficers))
        log.debugTag('currentNodes', len(self.currentNodes))
        self.swarm.clear()

        if self.freeOfficers and self.currentNodes:
            nodesId = [n.marker for n in self.currentNodes]
            self.cellIds = [o.id_ for o in self.freeOfficers] + nodesId
            self.solutionLen = len(self.cellIds)
            self.freeOfficersLen = len(self.freeOfficers)
            self.getCurrentEdges(nodesId)
            offisLeader = 0

            if len(self.freeOfficers) == 1:
                size = 1
            else:
                size = self.popuSize

            for i in range(size):
                if offisLeader == self.freeOfficersLen:
                    offisLeader = 0
                self.swarm.append(self.generateNewParticle(offisLeader))
                offisLeader += 1
            # delete variables from memory
            gc.collect()

    def generateNewParticle(self, offisLeader):
        agents = np.random.uniform(self.minX, self.maxX, size=self.solutionLen).tolist()
        agents[offisLeader] = -5000  # decide an officer to be a leader in a solution
        initPath = list(map(lambda x, y: [x, y], self.cellIds, agents))
        initPath.sort(key=lambda tup: tup[1])  # sort path by random number
        # self.localOptimisation(initPath)
        pro = self.computeAveragePro(initPath)
        particle = Particle(solution=initPath, pro=pro, pathLen=self.solutionLen)
        return particle

    def evolveParticles(self):
        if self.freeOfficersLen == 1:
            self.localOptimisation(self.swarm[0].pBestSolution)  # local optimisation
            self.gBestParticle = self.swarm[0]
        else:
            # get best particle of the population
            self.gBestParticle = max(self.swarm, key=attrgetter('pBestPro'))

            for i in range(self.maxGen):
                log.debugTag('evolveParticles', i)
                # The key param is a value that identifies the sorting property of the objects
                self.swarm.sort(key=lambda p: p.pBestPro, reverse=True)
                self.w = 0.5 + uniform(0, 1) / 2

                # log.getTime()
                for particle in self.swarm:
                    # compute new velocity of curr particle
                    for k in range(1, self.solutionLen):
                        r1 = uniform(0, 1)
                        r2 = uniform(0, 1)
                        if uniform(0, 1) > self.updateRate:
                            particle.velocity[k] = (self.w * particle.velocity[k]) + (
                                    self.c1 * r1 * (particle.pBestSolution[k][1] - particle.solution[k][1])) + (
                                                           self.c2 * r2 * (self.gBestParticle.pBestSolution[k][1] -
                                                                           particle.solution[k][1]))
                            particle.solution[k][1] += particle.velocity[k]
                            # adjust maximum position if necessary
                            if particle.solution[k][1] > self.maxX:
                                particle.solution[k][1] = self.maxX
                            # adjust minimum position if necessary
                            if particle.solution[k][1] < self.minX:
                                particle.solution[k][1] = self.minX

                    particle.solution.sort(key=lambda tup: tup[1])  # sort path by random number
                    if uniform(0, 1) > self.doLocRate:
                        self.localOptimisation(particle.solution)
                    particle.solutionPro = self.computeAveragePro(particle.solution)

                    # is new position a new best for the particle
                    if particle.solutionPro > particle.pBestPro:
                        particle.pBestPro = particle.solutionPro
                        particle.pBestSolution = copy.deepcopy(particle.solution)

                    # is new position a new best overall
                    if particle.solutionPro > self.gBestParticle.pBestPro:
                        self.gBestParticle.pBestPro = particle.solutionPro
                        self.gBestParticle.pBestSolution = copy.deepcopy(particle.solution)

                # log.timeDiffNow('swarm elitist')

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
            # O indicate that it is an officer Id, and the following ids are nodes that will be assigned to this
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

                    # sorting
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


# class that represents a particle
class Particle:

    def __init__(self, solution, pro, pathLen):
        self.solution = solution  # current solution
        self.pBestSolution = copy.deepcopy(solution)  # best solution (fitness) it has achieved so far
        self.solutionPro = pro
        self.pBestPro = pro  # set costs
        self.velocity = np.random.uniform(-1, 1, size=pathLen).tolist()  # velocity of a particle is a sequence

    def set_particle(self, newParticle):
        self.solution = newParticle.solution
        self.pBestSolution = newParticle.pBestSolution
        self.solutionPro = newParticle.solutionPro
        self.pBestPro = newParticle.pBestPro
        self.velocity = newParticle.velocity
