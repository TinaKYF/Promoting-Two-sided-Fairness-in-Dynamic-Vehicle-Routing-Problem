# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:58:20 2018

@author: Kai Qin
"""
import time

testTag = 'test: '
level = 1
startTime = 0


def debugTest(mes):
    print(testTag + str(mes))


def infoTest(mes):
    print(testTag + str(mes))


def debugTag(tag, mes):
    print(tag + ': ' + str(mes))


def infoTag(tag, mes):
    print(tag + ': ' + str(mes))


def iniTime():
    global startTime
    startTime = time.time()
    return startTime


def timeDiff(tag, t1):
    endTime = time.time()
    dif = str(endTime - t1)
    print(tag + ': ' + dif + 's')
    return dif


def timeDiffNow(tag):
    endTime = time.time()
    dif = str(endTime - startTime)
    print(tag + ': ' + dif + 's')
    return dif
