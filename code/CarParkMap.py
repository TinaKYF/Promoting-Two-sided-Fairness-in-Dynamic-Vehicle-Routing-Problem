# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:58 2018

@author: kyleq
"""
from Point import Point
import csv


class CarParkMap:
    """Car parking map"""

    def __init__(self):
        self.resultPath = None
        self.nodes = []
        self.officers = []
        self.edges = None
        self.vioHistories = None

    def initialMap(self, nodesData, disData, vioData, resultPath):
        """Initial map with edges and nodes"""
        self.resultPath = resultPath
        self.edges = disData
        self.vioHistories = vioData

        # create each node of two points
        for index, row in nodesData.iterrows():
            marker = row['st_marker_id']
            area=row['area']
            point = Point(marker,area, vioData[vioData['street_marker'] == marker])
            self.nodes.append(point)

    def getTravelTime(self, startPMarker, toPMarker, walkingSpeed):
        dis = self.getDistance(startPMarker, toPMarker)
        return dis / walkingSpeed

    def getDistance(self, startPMarker, toPMarker):
        if startPMarker == toPMarker:
            return 0

        nid1 = startPMarker + '_' + toPMarker
        nid2 = toPMarker + '_' + startPMarker
        condition = (self.edges.twoPs == nid1) | (self.edges.twoPs == nid2)
        e = self.edges[condition]
        if len(e) == 0:
            return None
        else:
            return e.iloc[0]['distance']

    def findEdgeByNodes(self, twoPsList):
        e = self.edges[self.edges.twoPs.isin(twoPsList)]
        if len(e) == 0:
            return None
        else:
            return e

    def findOfficerById(self, id_):
        for o in self.officers:
            if o.id_ == id_:
                return o
        return None

    def findNodeByMarkerId(self, markerId):
        for node in self.nodes:
            if node.marker == markerId:
                return node
        return None

    def findNodeByMarkers(self, markers):
        nodes = [node for node in self.nodes if node.marker in markers]
        return nodes

    def getFreeOfficersNum(self):
        free = 0
        for officer in self.officers:
            if officer.assigned is False:
                free += 1
        return free

    def getFreeOfficers(self):
        officers = [officer for officer in self.officers if officer.assigned is False]
        return officers

    def getnotcountVioNodes(self, curTime):
        nodes = [node for node in self.nodes if node.hasfindvio(curTime) is True]
        return nodes

    def findClosestFreeOfficer(self, nextPMarker):
        minDis = 1000000
        closestOfficer = None
        for officer in self.getFreeOfficers():
            if officer.assigned is False:
                if officer.occupiedMarker == nextPMarker:
                    dis = 0
                else:
                    dis = self.getDistance(officer.occupiedMarker, nextPMarker)

                if minDis > dis:
                    minDis = dis
                    closestOfficer = officer
        return closestOfficer

    def getFreeVioNodes(self, curTime):
        nodes = [node for node in self.nodes if node.assigned is False and node.hasViolation(curTime) is True]
        return nodes

    def getFreeVioNodesId(self, curTime):
        nodesId = [node.marker for node in self.nodes if node.assigned is False and node.hasViolation(curTime) is True]
        return nodesId

    def releaseNode(self, markerId):
        node = self.findNodeByMarkerId(markerId)
        node.assigned = False

    def printAllResults(self, fileName, titles, params):
        pathLenTotal = 0
        benefitTotal = 0
        saveTimeTotal = 0
        distanceTotal = 0
        validDistance = 0

        for officer in self.officers:
            pathLenTotal += len(officer.myPath)
            benefitTotal += officer.benefit
            saveTimeTotal += officer.saveTime
            distanceTotal += officer.totalDis
            validDistance += officer.validDis
            officer.showResult()
        titles = titles + ['officersNum', 'pathTotalSize', 'benefits', 'spareTime', 'totalWalkDis', 'validDistance']
        params = params + [len(self.officers), pathLenTotal, benefitTotal, saveTimeTotal, distanceTotal, validDistance]
        results = [titles, params]
        # results = [params]
        with open(self.resultPath + fileName, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(results)
