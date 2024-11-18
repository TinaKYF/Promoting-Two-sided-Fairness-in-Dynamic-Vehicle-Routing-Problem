# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:58 2018

@author: kyleq
"""
from Officer import Officer
import Log as log
import datetime
import copy
import statistics


class GreedyAlgorithm1:
    """
    In TOP or MTOP, a vio node will be assigned to the officer who has the highest probability to collect fines.
    The vio node will be scanned in order in the list.
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
        fileName = 'greedy1_result.csv'
        titles = ['runTime', 'stopTime', 'startTime', 'endTime','officerdis_record', 'stdofficerdis_record', 'area_vionum', 'area_capnum', 'area_caprate',
                  'stdarea_caprate']
        params = [runTime, stopTime, self.startTime, self.endTime,officerdis_record, stdofficerdis_record, self.area_list, self.area_cap_list,
                  areacaprate, stdareacaprate]
        self.carParkMap.printAllResults(fileName, titles, params)

    def updateOfficersStatus(self):
        for officer in self.carParkMap.officers:
            # when an officer finish his traveling
            if officer.assigned is True and officer.arriveTime <= self.currentTime:
                officer.assigned = False
                officer.nextIntendedPoint = None
                self.carParkMap.releaseNode(officer.occupiedMarker)

    def assignNextPToOfficers(self):
        self.currentcountNodes = self.carParkMap.getnotcountVioNodes(self.currentTime)
        if self.currentcountNodes:
            for node in self.currentcountNodes:
                self.area_list[self.area_dic[node.area]] += 1
                node.removecountRecord(self.currentTime)
        freeMenNum = self.carParkMap.getFreeOfficersNum()
        if freeMenNum > 0:  # if there are free officers this time
            assignedNum = 0
            # find current violation records at current time
            for index, row in self.carParkMap.vioHistories.iterrows():
                if assignedNum == freeMenNum:  # if all the officers have been assigned
                    break

                if row['vioTime'] < self.currentTime < row['depTime']:
                    vioTime = row['vioTime']
                    marker = row['street_marker']
                    node = self.carParkMap.findNodeByMarkerId(marker)
                    self.freeOfficers = self.carParkMap.getFreeOfficers()
                    # the parking violation is happening
                    if node.assigned is False:
                        officer = self.getOfficerWithMaxPro(marker, vioTime)
                        dis = self.carParkMap.getDistance(officer.occupiedMarker, marker)
                        travelTime = dis / officer.walkingSpeed
                        officer.arriveTime = self.currentTime + travelTime
                        officer.myPath.append(marker)
                        officer.occupiedMarker = marker
                        officer.assigned = True
                        officer.totalDis += dis
                        node.assigned = True  # update node's status
                        assignedNum += 1

                        # benefit increase when officer arrive before leaving for this record
                        if officer.arriveTime < row['depTime']:
                            officer.benefit += 1
                            officer.validDis += dis
                            self.totalBenefit += 1
                            self.area_cap_list[self.area_dic[node.area]] += 1

                        # remove this record from record history
                        self.carParkMap.vioHistories.drop(index, inplace=True)

            # if there are officers who have no assignment after scanning records at this time
            for officer in self.carParkMap.officers:
                if officer.assigned is False:
                    officer.saveTime += self.updateTime

    def getOfficerWithMaxPro(self, nodeMarker, vioTime):
        maxPro = -2
        winnerOffi = None
        proLen = len(self.stayProList)

        for officer in self.freeOfficers:
            if officer.assigned is False:
                travelTime = self.carParkMap.getTravelTime(officer.occupiedMarker, nodeMarker,
                                                           officer.walkingSpeed)
                arriveTime = self.currentTime + travelTime
                proIndex = arriveTime - vioTime
                if proIndex >= proLen:
                    probability = 0
                else:
                    probability = self.stayProList[int(proIndex)]

                if probability > maxPro:
                    maxPro = probability
                    winnerOffi = officer
        return winnerOffi
