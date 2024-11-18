# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:19 2018

@author: kyleq
"""


class Officer:
    """Officer to collect fines"""

    def __init__(self, id_, walkingSpeed=70, startPointMarker='', area=''):
        self.id_ = 'O' + str(id_)  # O indicate that its a officer Id
        self.walkingSpeed = walkingSpeed
        self.area = area
        self.occupiedMarker = startPointMarker
        self.assigned = False
        self.arriveTime = 0
        self.myPath = []
        self.myNextIntendedPoints = []  # [{"nextP": marker, "recordId": recordId, "probability": 0.6}]
        self.nextIntendedPoint = None  # {"nextP": marker0, "recordId": recordId0, "probability": max}
        self.benefit = 0
        self.saveTime = 0
        self.totalDis = 0
        self.validDis = 0

    def getNextPWithMaxProb(self):
        pro = -1.0
        self.nextIntendedPoint = None
        for row in self.myNextIntendedPoints:
            if row["probability"] > pro:
                pro = row["probability"]
                self.nextIntendedPoint = row

    def deleteConflictIntendedPoint(self, conflictPMarker):
        for point in self.myNextIntendedPoints:
            if point["nextP"] == conflictPMarker:
                self.myNextIntendedPoints.remove(point)
                break

    def showResult(self):
        print("---------------------------------------------")
        print('Officer Id:' + str(self.id_))
        print('benefit:' + str(self.benefit))
        print('save time:' + str(self.saveTime))
        print('path size:' + str(len(self.myPath)))
        print('travel path:')
        print(self.myPath)
        print("---------------------------------------------")
