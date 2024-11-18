# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:15:58 2018

@author: kyleq
"""

import copy
class Point:
    """ Point on map for each parking slot """

    def __init__(self, marker,area, vioRecords=None):
        self.marker = marker
        self.vioRecords = vioRecords
        self.countRecords = copy.deepcopy(vioRecords)
        self.assigned = False
        self.area=area
        # self.count=False

    def calculateProbability(self, arriveTime, curTime, stayProList):
        """ Calculate leveling probability for a next arriving point """
        probability = -1
        recordId = None
        proLen = len(stayProList)

        for index, row in self.vioRecords.iterrows():
            vioTime = row['vioTime']
            depTime = row['depTime']
            if vioTime < curTime < depTime:
                recordId = index
                proIndex = arriveTime - vioTime
                if proIndex >= proLen:
                    probability = 0
                else:
                    probability = stayProList[int(proIndex)]
                break
        return probability, recordId

    def hasViolation(self, curTime):
        for index, row in self.vioRecords.iterrows():
            if row['vioTime'] < curTime < row['depTime']:
                return True
        return False

    def hasfindvio(self, curTime):
        for index, row in self.countRecords.iterrows():
            if row['vioTime'] < curTime < row['depTime']:
                return True
        return False

    def getDepTimeByRecordId(self, recordId):
        r = self.vioRecords.loc[recordId]
        return r['depTime']

    def getdeptime(self,curTime):
        for index, row in self.vioRecords.iterrows():
            if row['vioTime'] < curTime < row['depTime']:
                depTime = row['depTime']
                return depTime
        return None

    def removeRecordByTime(self, curTime):
        for index, row in self.vioRecords.iterrows():
            if row['vioTime'] < curTime < row['depTime']:
                depTime = row['depTime']
                self.vioRecords.drop(index, inplace=True)
                return depTime
        return None

    def removecountRecord(self, curTime):
        for index, row in self.countRecords.iterrows():
            if row['vioTime'] < curTime < row['depTime']:
                self.countRecords.drop(index, inplace=True)
        return None

    def removeRecordById(self, recordId):
        self.vioRecords.drop(recordId, inplace=True)
