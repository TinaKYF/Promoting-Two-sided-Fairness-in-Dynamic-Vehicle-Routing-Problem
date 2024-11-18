# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:58 2018

@author: kyleq
"""
from Officer import Officer
from random import uniform
from random import randint
from operator import attrgetter
from pandas import Series
import numpy as np
import Log as log
import math
import copy
import gc
import csv
import datetime
import statistics

class PSOAlgorithm2:
    """
    Tow-part chromosome
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
        self.particles = []
        self.solutionPathLength = 0
        self.solutionOfficesNum = 0
        self.popuSize = 100
        self.maxGen = 300
        self.w = 1.1  # inertia
        self.c1 = 2  # cognitive (particle)
        self.c2 = 2  # social (swarm)
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
        fileName = 'PSO2_result.csv'
        titles = ['runTime', 'stopTime', 'startTime', 'endTime', 'updateTime', 'popuSize', 'w', 'c1', 'c2',
                  'pointNumLoc', 'doLocRate', 'maxGen', 'proThreshold', 'validCapturedTime','officerdis_record', 'stdofficerdis_record', 'area_vionum', 'area_capnum', 'area_caprate',
                  'stdarea_caprate']
        params = [runTime, stopTime, self.startTime, self.endTime, self.updateTime, self.popuSize, self.w, self.c1,
                  self.c2, self.pointNumLoc, self.doLocRate, self.maxGen, self.proThreshold, self.validCapturedTime,officerdis_record, stdofficerdis_record,self.area_list, self.area_cap_list,
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
        if self.particles:
            self.evolveParticles()

            solutionBest = self.gBestParticle.pBestSolution
            part1Solution = solutionBest[0]
            part2Solution = solutionBest[1]
            part1Index = 0
            preBreak = 1

            for i in range(self.solutionOfficesNum):  # iterations for free officers
                officer = self.freeOfficers[i]
                curBreak = part2Solution[i]
                assignNum = curBreak - preBreak

                if assignNum > 0:
                    nextPMarker = part1Solution[part1Index]  # only the first node will be assigned in real work
                    part1Index += assignNum  # move to next officer's first-assigned node

                    # update next position for officer
                    dis = self.getDistance(officer.occupiedMarker, nextPMarker)
                    travelTime = dis / officer.walkingSpeed
                    officer.arriveTime = self.currentTime + travelTime
                    officer.myPath.append(nextPMarker)
                    officer.occupiedMarker = nextPMarker
                    officer.assigned = True
                    officer.totalDis += dis

                    # update node's record status
                    node = self.getCurrentNodeById(nextPMarker)
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
                preBreak = curBreak
            # results = [[self.currentTime, self.totalBenefit]]
            # with open('70_offi_2_uptime_genetic_solution_result.csv', "a") as output:
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
        self.particles.clear()

        if self.freeOfficers and self.currentNodes:
            vioNodesId = [n.marker for n in self.currentNodes]
            freeOfficersId = [o.id_ for o in self.freeOfficers]
            self.getCurrentEdges(vioNodesId)
            self.solutionPathLength = len(vioNodesId)
            self.solutionOfficesNum = len(freeOfficersId)
            seqNum = int(self.solutionPathLength / 2)

            if self.solutionOfficesNum == 1:
                size = 1
            else:
                size = self.popuSize

            for i in range(size):
                solution = [self.fisherYatesShuffle(copy.deepcopy(vioNodesId)), self.generateSolutionBreaks()]
                # self.localOptimisation(solution)  # local optimisation
                prop = self.computeAveragePro(solution)

                # generate initial velocity
                velocity = []
                for j in range(seqNum):
                    a = randint(1, self.solutionPathLength)
                    b = randint(1, self.solutionPathLength)
                    while a == b:
                        a = randint(1, self.solutionPathLength)
                        b = randint(1, self.solutionPathLength)
                    velocity.append([a, b])

                particle = Particle(solution=solution, pro=prop, velocity=velocity)
                self.particles.append(particle)
            # delete variables from memory
            del vioNodesId
            del freeOfficersId
            gc.collect()

    def getNodeIndexInSolution(self, nodeId, path):
        index = [i for i, ele in enumerate(path) if ele == nodeId]
        return index[0]

    def roundUpSequence(self, num):
        num = math.ceil(num)
        if num < 1:
            num = 1
        elif num > self.solutionPathLength:
            num = self.solutionPathLength
        return num

    def roundUpBreak(self, num):
        num = math.ceil(num)
        if num < 1:
            num = 1
        elif num > self.solutionPathLength + 1:
            num = self.solutionPathLength + 1
        return num

    def sequenceSimilarityCal(self, sequence1, sequence2):
        sequence3 = []
        longSeqLen = len(sequence1)
        shortSeqLen = len(sequence2)

        # find a longer sequence to iterate`
        if longSeqLen >= shortSeqLen:
            longSequence = sequence1
            shortSequence = sequence2
        else:
            longSequence = sequence2
            shortSequence = sequence1
            longSeqLen, shortSeqLen = shortSeqLen, longSeqLen

        for i in range(longSeqLen):
            seq1Ele = longSequence[i]
            if i < shortSeqLen:
                seq2Ele = shortSequence[i]
                sequence3.append(
                    [self.roundUpSequence(seq1Ele[0] + (seq2Ele[0] - seq1Ele[0]) / 2),
                     self.roundUpSequence(seq1Ele[1] + (seq2Ele[1] - seq1Ele[1]) / 2)])
            else:
                sequence3.append(seq1Ele)
        return sequence3

    def breakSimilarityCal(self, pBestBreaks, gBestBreaks, currentBreaks):
        assignAll = False
        preBreak = 1
        breaks = []

        for i in range(self.solutionOfficesNum):
            if assignAll:
                breaks.append(self.solutionPathLength)
            elif i == (self.solutionOfficesNum - 1):
                breaks.append(self.solutionPathLength)
            else:
                _continue = True
                # loop util get a suitable break
                while _continue:
                    r3 = uniform(0, 1)
                    r4 = uniform(0, 1)
                    a = r3 * (pBestBreaks[i] + (currentBreaks[i] - pBestBreaks[i]) / 2)
                    b = r4 * (gBestBreaks[i] + (currentBreaks[i] - gBestBreaks[i]) / 2)
                    if a > 0:
                        a = self.roundUpBreak(a)
                    if b > 0:
                        b = self.roundUpBreak(b)

                    c = a + (b - a) / 2
                    if c > 0:
                        c = self.roundUpBreak(c)
                    curBreak = c

                    if self.solutionPathLength > curBreak >= preBreak:
                        breaks.append(curBreak)
                        preBreak = curBreak
                        break
                    elif curBreak == self.solutionPathLength:
                        breaks.append(curBreak)
                        preBreak = curBreak
                        assignAll = True
                        break
        return breaks

    def evolveParticles(self):
        if self.solutionOfficesNum == 1:
            self.localOptimisation(self.particles[0].pBestSolution)  # local optimisation
            self.gBestParticle = self.particles[0]
        else:
            # updates best particle of the population
            self.gBestParticle = max(self.particles, key=attrgetter('pBestPro'))

            for t in range(self.maxGen):
                log.debugTag('evolveParticles2', t)

                # for each particle in the swarm
                for particle in self.particles:
                    r1 = uniform(0, 1)
                    r2 = uniform(0, 1)
                    sequences1 = []
                    sequences2 = []
                    gBestSol = copy.deepcopy(self.gBestParticle.pBestSolution)  # gets solution of the gbest
                    pBestSol = copy.deepcopy(particle.pBestSolution)  # copy of the pbest solution
                    currentSol = copy.deepcopy(particle.solution)  # gets copy of the current solution
                    currentVel = copy.deepcopy(particle.velocity)  # gets copy of the current particle velocity

                    for i in range(self.solutionPathLength):
                        if currentSol[0][i] != pBestSol[0][i]:
                            a = i
                            b = self.getNodeIndexInSolution(currentSol[0][i], pBestSol[0])
                            # append swap sequence in the list of velocity
                            sequences1.append([self.roundUpSequence(self.c1 * r1 * (a + 1)),
                                               self.roundUpSequence(self.c1 * r1 * (b + 1))])
                            # makes the swap random key
                            pBestSol[0][a], pBestSol[0][b] = pBestSol[0][b], pBestSol[0][a]

                        if currentSol[0][i] != gBestSol[0][i]:
                            a = i
                            b = self.getNodeIndexInSolution(currentSol[0][i], gBestSol[0])
                            # append swap operator in the list of velocity
                            sequences2.append([self.roundUpSequence(self.c2 * r2 * (a + 1)),
                                               self.roundUpSequence(self.c2 * r2 * (b + 1))])
                            # makes the swap random key
                            gBestSol[0][a], gBestSol[0][b] = gBestSol[0][b], gBestSol[0][a]

                    for seq in currentVel:
                        seq[0] = self.roundUpSequence(seq[0] * self.w)
                        seq[1] = self.roundUpSequence(seq[1] * self.w)

                    # updates velocity
                    sequences3 = self.sequenceSimilarityCal(sequences1, sequences2)
                    particle.velocity = self.sequenceSimilarityCal(currentVel, sequences3)

                    # generates new path for particle
                    for swapSeq in particle.velocity:
                        a = swapSeq[0] - 1
                        b = swapSeq[1] - 1
                        # makes the swap
                        currentSol[0][a], currentSol[0][b] = currentSol[0][b], currentSol[0][a]

                    if self.solutionOfficesNum > 1:
                        currentSol[1] = self.breakSimilarityCal(pBestSol[1], gBestSol[1], currentSol[1])
                    if uniform(0, 1) > self.doLocRate:
                        self.localOptimisation(currentSol)
                    particle.solution = currentSol
                    particle.solutionPro = self.computeAveragePro(currentSol)

                    # checks if current solution is pbest solution
                    if particle.solutionPro > particle.pBestPro:
                        particle.pBestSolution = copy.deepcopy(particle.solution)
                        particle.pBestPro = particle.solutionPro

                    # checks if current solution is gbest solution
                    if particle.solutionPro > self.gBestParticle.pBestPro:
                        self.gBestParticle.pBestSolution = copy.deepcopy(particle.solution)
                        self.gBestParticle.pBestPro = particle.solutionPro

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

    def computeAveragePro(self, solution):
        propTotal = 0.0
        assumedCapturedNum = 0
        part1Solution = solution[0]
        part2Solution = solution[1]
        part1Index = 0
        preBreak = 1  # In order to calculate new breaks, add 1 for a break

        for i, curBreak in enumerate(part2Solution):  # iterate each officer's assigned nodes
            officer = self.freeOfficers[i]
            currentNodeId = officer.occupiedMarker
            assumedCurTime = self.currentTime
            assignNum = curBreak - preBreak

            for j in range(assignNum):  # could be 0, loop will skip
                # concern on next 3 points only
                if j >= self.pointNumLoc:
                    part1Index += (assignNum - self.pointNumLoc)
                    break

                nextNode = self.getCurrentNodeById(part1Solution[part1Index])
                dis = self.getDistance(currentNodeId, nextNode.marker)
                if dis is None:
                    break
                travelTime = dis / officer.walkingSpeed
                arriveTime = assumedCurTime + travelTime
                probability, recordId = nextNode.calculateProbability(arriveTime, assumedCurTime, self.stayProList)
                assumedCurTime = arriveTime
                currentNodeId = nextNode.marker

                if probability > self.proThreshold:
                    propTotal += probability
                    assumedCapturedNum += 1
                part1Index += 1
            preBreak = curBreak
        if assumedCapturedNum == 0:
            return 0
        else:
            return propTotal / assumedCapturedNum

    def localOptimisation(self, solution):
        """
        Reorder points in each sub path with max pro from the previous node
        :param solution:
        """
        pathIndex = 0
        part1Solution = solution[0]
        part2Solution = solution[1]
        preBreak = 1  # In order to calculate new breaks, add 1 for a break

        for i, curBreak in enumerate(part2Solution):  # iterate each officer's assigned nodes
            officer = self.freeOfficers[i]
            currentNodeId = officer.occupiedMarker
            assumedCurTime = self.currentTime
            assignNum = curBreak - preBreak
            pointsCount = 0

            if assignNum > 1:
                subPathIds = [part1Solution[pathIndex + j] for j in range(assignNum)]
                while len(subPathIds) > 1:
                    # concern about next 3 points only
                    if pointsCount >= self.pointNumLoc:
                        # move index in path to next officer
                        pathIndex += (assignNum - self.pointNumLoc)
                        break
                    pointsCount += 1

                    # sorting
                    proMax = -2
                    winner = None
                    for pId in subPathIds:
                        nextNode = self.getCurrentNodeById(pId)
                        travelTime = self.getDistance(currentNodeId, pId) / officer.walkingSpeed
                        arriveTime = assumedCurTime + travelTime
                        pro, recordId = nextNode.calculateProbability(arriveTime, assumedCurTime, self.stayProList)
                        if pro > proMax:
                            proMax = pro
                            winner = [pId, arriveTime]
                    wIndex = part1Solution.index(winner[0])
                    currentNodeId = winner[0]  # update current node with winner node
                    assumedCurTime = winner[1]  # update current time
                    if pathIndex != wIndex:  # update the winnerNodeId to original path in order
                        part1Solution[pathIndex], part1Solution[wIndex] = part1Solution[wIndex], part1Solution[
                            pathIndex]
                    pathIndex += 1
                    subPathIds.remove(currentNodeId)

                    if len(subPathIds) == 1:  # update the last point in subPathIds to original path
                        part1Solution[pathIndex] = subPathIds[0]
                        pathIndex += 1
                        subPathIds.clear()
            else:
                pathIndex += assignNum
            preBreak = curBreak

    def fisherYatesShuffle(self, path):
        """
        Shuffle a sample randomly and return.
        :param path:
        :return: path
        """
        list_range = range(0, len(path))
        for i in list_range:
            j = randint(list_range[0], list_range[-1])
            path[i], path[j] = path[j], path[i]
        return path

    def generateSolutionBreaks(self):
        """
        In order to calculate breaks in S1 . S2 = S1 + (S2 − S1) ÷ 2, set 0 node as 1 (add 1) for a break,
        otherwise the new break will be always 0 after calculation
        :return: breaks list
        """
        assignNums = []
        breaks = []

        if self.solutionOfficesNum == 1:  # assign all nodes if there is one office
            breaks.append(self.solutionPathLength)
        else:
            # initialise part 2 for current free officers who could have 0 next node, 0 == 1 for calculation need
            for i in range(self.solutionOfficesNum):
                assignNums.append(1)

            # pick officers for each vio node
            for i in range(self.solutionPathLength):
                index = randint(0, self.solutionOfficesNum - 1)
                assignNums[index] += 1

            preBreak = 1
            for n in assignNums:
                curBreak = n + preBreak - 1
                breaks.append(curBreak)
                preBreak = curBreak
        return breaks


# class that represents a particle
class Particle:

    def __init__(self, solution, pro, velocity):
        # current solution
        self.solution = solution

        # best solution (fitness) it has achieved so far
        self.pBestSolution = copy.deepcopy(solution)

        # set costs
        self.solutionPro = pro
        self.pBestPro = pro

        # velocity of a particle is a sequence of 2-tuple
        # (1, 2) means SO(1,2)
        self.velocity = copy.deepcopy(velocity)
