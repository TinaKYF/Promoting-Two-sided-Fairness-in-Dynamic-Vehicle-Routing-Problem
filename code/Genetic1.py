# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:58 2018

@author: kyleq
"""
from Officer import Officer
from random import uniform
from random import randint
from pandas import Series
import Log as log
import copy
import gc
import datetime
import statistics

class GeneticAlgorithm:
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
        self.populations = []
        self.popuSize = 100
        self.part1ChromLength = 0
        self.part2ChromLength = 0
        self.crossRate = 0.9
        self.mutateRate = 0.3
        self.replacePercent = 0.4
        self.pointNumLoc = 3
        self.doLocRate = 0.5
        self.maxGen = 300
        self.proThreshold = 0.21
        self.processWay = 1  # 1 - more free officers than vio nodes;2 - more vio nodes than free officers;3 - equal
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
        fileName = 'genetic_result.csv'
        titles = ['runTime', 'stopTime', 'startTime', 'endTime', 'updateTime', 'popuSize', 'crossRate', 'mutateRate',
                  'replacePercent', 'pointNumLoc', 'doLocRate', 'maxGen', 'proThreshold','officerdis_record', 'stdofficerdis_record', 'area_vionum', 'area_capnum', 'area_caprate',
                  'stdarea_caprate']
        params = [runTime, stopTime, self.startTime, self.endTime, self.updateTime, self.popuSize, self.crossRate,
                  self.mutateRate, self.replacePercent, self.pointNumLoc, self.doLocRate, self.maxGen,
                  self.proThreshold,officerdis_record, stdofficerdis_record,self.area_list, self.area_cap_list,
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
        if len(self.populations) > 0:
            self.evolvePopulation()

            self.populations.sort(key=lambda tup: -tup[1])
            chroBest = self.populations[0][0]
            part1ChroBest = chroBest[0]
            part2ChroBest = chroBest[1]

            part1ChroIndex = 0
            for i in range(self.part2ChromLength):  # iterations for free officers
                assignedNodeNum = part2ChroBest[i]
                officer = self.freeOfficers[i]

                if assignedNodeNum > 0:
                    nextPMarker = part1ChroBest[part1ChroIndex]  # only the first node will be assigned in real work
                    part1ChroIndex += assignedNodeNum  # move to next officer's first-assigned node

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
                        self.area_cap_list[self.area_dic[node.area]] += 1
                        self.totalBenefit += 1
                else:
                    officer.saveTime += self.updateTime
        else:
            for officer in self.freeOfficers:
                officer.saveTime += self.updateTime
        # results = [[self.currentTime, self.totalBenefit]]
        # with open('25_offi_genetic_solution_result.csv', "a") as output:
        #     writer = csv.writer(output, lineterminator='\n')
        #     writer.writerows(results)

    def initialPopulations(self):
        self.currentcountNodes = self.carParkMap.getnotcountVioNodes(self.currentTime)
        if self.currentcountNodes:
            for node in self.currentcountNodes:
                self.area_list[self.area_dic[node.area]] += 1
                node.removecountRecord(self.currentTime)
        self.freeOfficers = self.carParkMap.getFreeOfficers()
        self.currentNodes = self.carParkMap.getFreeVioNodes(self.currentTime)
        self.populations.clear()
        log.debugTag('freeOfficers', len(self.freeOfficers))
        log.debugTag('currentNodes', len(self.currentNodes))

        if self.freeOfficers and self.currentNodes:
            vioNodesId = [n.marker for n in self.currentNodes]
            freeOfficersId = [o.id_ for o in self.freeOfficers]
            self.getCurrentEdges(vioNodesId)
            self.part1ChromLength = len(vioNodesId)
            self.part2ChromLength = len(freeOfficersId)

            if len(self.freeOfficers) == 1:
                size = 1
            else:
                size = self.popuSize

            for i in range(size):
                chromosome = [self.fisherYatesShuffle(copy.deepcopy(vioNodesId)),
                              self.generatePart2Chromosome()]
                # self.localOptimisation(chromosome)  # local optimisation
                propTotal = self.computeAveragePro(chromosome)
                self.populations.append([chromosome, propTotal])
            # delete variables from memory
            del vioNodesId
            del freeOfficersId
            gc.collect()

    def evolvePopulation(self):
        if self.part2ChromLength == 1:
            self.localOptimisation(self.populations[0][0])  # local optimisation
        else:
            for t in range(self.maxGen):
                log.debugTag('evolvePopulation', t)

                # generate new populations
                replaceNum = int(self.replacePercent * self.popuSize)
                for i in range(replaceNum):
                    a = -1
                    b = -1
                    # Select parents, a and b must be different
                    while a == b:
                        a = self.selectByRouletteWheel()
                        b = self.selectByRouletteWheel()

                    parent1Chro = self.populations[a][0]
                    parent2Chro = self.populations[b][0]
                    child1Chro, child2Chro = None, None

                    if self.part2ChromLength > 1:  # cross and mutate when more vio nodes than free officers
                        # Possibly do a crossover
                        if uniform(0, 1) < self.crossRate:
                            child1Chro, child2Chro = self.twoPartsCrossover(parent1Chro, parent2Chro)

                        if uniform(0, 1) < self.mutateRate:
                            if child1Chro is not None:
                                child1Chro = self.swapMutation(child1Chro)
                                child2Chro = self.swapMutation(child2Chro)
                            else:
                                child1Chro = self.swapMutation(parent1Chro)
                                child2Chro = self.swapMutation(parent2Chro)
                    else:
                        child1Chro = self.swapMutation(parent1Chro)
                        child2Chro = self.swapMutation(parent2Chro)

                    if child1Chro is not None:  # if they have children, do replacement
                        if uniform(0, 1) < self.doLocRate:
                            self.localOptimisation(child1Chro)  # local optimisation
                        if uniform(0, 1) < self.doLocRate:
                            self.localOptimisation(child2Chro)  # local optimisation
                        child1Pro = self.computeAveragePro(child1Chro)
                        child2Pro = self.computeAveragePro(child2Chro)
                        popus = [self.populations[a], self.populations[b], [child1Chro, child1Pro],
                                 [child2Chro, child2Pro]]
                        popus.sort(key=lambda tup: -tup[1])
                        self.populations[a] = popus[0]
                        self.populations[b] = popus[1]

    def selectByRouletteWheel(self):
        """
        This method implements the roulette wheel selection.
        :return: the index of the selected member (-1 if it goes wrong)
        """
        totalProSum = sum(row[1] for row in self.populations)
        rand = uniform(0, totalProSum)
        partialSum = 0
        for i in range(self.popuSize):
            partialSum += self.populations[i][1]
            if partialSum >= rand:
                return i
        return -1

    def twoPartsCrossover(self, chro1, chro2):
        child1Chro = self.generateChildChro(chro1, chro2)
        child2Chro = self.generateChildChro(chro2, chro1)
        return child1Chro, child2Chro

    def generateChildChro(self, momChro, dadChro):
        segmentSide = []  # gene segment num for each officer
        savedGenesPool = []
        part1ChroIndex = 0
        child1Chro = copy.deepcopy(momChro)

        for i in range(self.part2ChromLength):  # the number of free officers
            assignedNodeNum = momChro[1][i]
            if assignedNodeNum == 0:
                segmentSide.append(0)
                continue

            segmentSide.append(randint(1, assignedNodeNum))  # selected segment num for a officer
            if assignedNodeNum > segmentSide[i]:
                startSegmentIndex = randint(1, assignedNodeNum - segmentSide[i]) - 1
                startIndex = part1ChroIndex + startSegmentIndex  # start index for an officer's assigned nodes
                for k in range(segmentSide[i]):  # copy selected genes into savedGenesPool
                    savedGenesPool.append(momChro[0][startIndex + k])
            else:
                for k in range(segmentSide[i]):  # copy all gene into savedGenesPool
                    savedGenesPool.append(momChro[0][part1ChroIndex + k])
            part1ChroIndex += assignedNodeNum  # go to next officer's owning nodes

        # get left genes according to the first part of Dad’s chromosome
        unSavedGenesPool = [marker for marker in dadChro[0] if marker not in savedGenesPool]

        part1ChroIndex = 0
        for i in range(self.part2ChromLength):  # the number of free officers
            if len(savedGenesPool) > 0:
                # copy first genes segment to child chro for each officer
                for k in range(segmentSide[i]):
                    child1Chro[0][part1ChroIndex] = savedGenesPool.pop(0)
                    part1ChroIndex += 1

            unSavedLen = len(unSavedGenesPool)
            if unSavedLen > 0:  # if still having unSaved points left
                if i != self.part2ChromLength - 1:  # not the last man
                    segmentSide2 = randint(1, unSavedLen)
                else:
                    segmentSide2 = unSavedLen

                # copy second genes segment to child chro for each officer
                for k in range(segmentSide2):
                    child1Chro[0][part1ChroIndex] = unSavedGenesPool.pop(0)
                    part1ChroIndex += 1
                # add up two segments' num for part2
                segmentSide[i] += segmentSide2
        child1Chro[1] = segmentSide
        return child1Chro

    def swapMutation(self, parentChro):
        # swap part1
        part1Chro = parentChro[0].copy()
        if len(part1Chro) > 1:
            r1 = randint(0, self.part1ChromLength - 1)
            r2 = randint(0, self.part1ChromLength - 1)
            while r1 == r2:
                r2 = randint(0, self.part1ChromLength - 1)
            # swap array elements at those indices
            part1Chro[r1], part1Chro[r2] = part1Chro[r2], part1Chro[r1]

        # swap part2
        part2Chro = parentChro[1].copy()
        if len(part2Chro) > 1:
            k1 = randint(0, self.part2ChromLength - 1)
            k2 = randint(0, self.part2ChromLength - 1)
            while k1 == k2:
                k2 = randint(0, self.part2ChromLength - 1)
            # swap array elements at those indices
            part2Chro[k1], part2Chro[k2] = part2Chro[k2], part2Chro[k1]

        childChro = [part1Chro, part2Chro]
        return childChro

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

    def computeAveragePro(self, chromosome):
        propTotal = 0.0
        assumedCapturedNum = 0
        part1Chro = chromosome[0]
        part2Chro = chromosome[1]
        part1Index = 0

        for i, gene in enumerate(part2Chro):
            officer = self.freeOfficers[i]
            currentNodeId = officer.occupiedMarker
            assumedCurTime = self.currentTime

            for j in range(gene):  # gene could be 0, loop will skip
                # concern on next 3 points only
                if j >= self.pointNumLoc:
                    part1Index += (gene - self.pointNumLoc)
                    break

                nextNode = self.getCurrentNodeById(part1Chro[part1Index])
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
        if assumedCapturedNum == 0:
            return 0
        else:
            return propTotal / assumedCapturedNum

    def localOptimisation(self, chromosome):
        """
        Reorder points in each sub path with max pro from the previous node
        :param chromosome: chromosome
        """
        pathIndex = 0
        part1Chro = chromosome[0]
        part2Chro = chromosome[1]

        for i, gene in enumerate(part2Chro):
            officer = self.freeOfficers[i]
            currentNodeId = officer.occupiedMarker
            assumedCurTime = self.currentTime
            pointsCount = 0

            if gene > 1:
                subPathIds = [part1Chro[pathIndex + j] for j in range(gene)]
                while len(subPathIds) > 1:
                    # concern on next 3 points only
                    if pointsCount >= self.pointNumLoc:
                        # move index in path to next officer
                        pathIndex += (gene - self.pointNumLoc)
                        break
                    pointsCount += 1
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
                    wIndex = part1Chro.index(winner[0])
                    currentNodeId = winner[0]  # update current node with winner node
                    assumedCurTime = winner[1]  # update current time
                    if pathIndex != wIndex:  # update the winnerNodeId to original path in order
                        part1Chro[pathIndex], part1Chro[wIndex] = part1Chro[wIndex], part1Chro[pathIndex]
                    pathIndex += 1
                    subPathIds.remove(currentNodeId)

                    if len(subPathIds) == 1:  # update the last point in subPathIds to original path
                        part1Chro[pathIndex] = subPathIds[0]
                        pathIndex += 1
                        subPathIds.clear()
            else:
                pathIndex += gene

    def fisherYatesShuffle(self, chro):
        """
        Shuffle a sample randomly and return.
        :param chro:
        :return:
        """
        list_range = range(0, len(chro))
        for i in list_range:
            j = randint(list_range[0], list_range[-1])
            chro[i], chro[j] = chro[j], chro[i]
        return chro

    def generatePart2Chromosome(self):
        chro = []

        if self.part2ChromLength == 1:  # assign all nodes if there is one office
            chro.append(self.part2ChromLength)
        else:
            # initialise part 2 for current free officers who could have 0 next node
            for i in range(self.part2ChromLength):
                chro.append(0)

            # pick officers for each vio node
            for i in range(self.part1ChromLength):
                index = randint(0, self.part2ChromLength - 1)
                chro[index] += 1
        return chro
