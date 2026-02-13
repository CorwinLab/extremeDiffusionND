from getHistogramStats import calculateStatistics
import sys

if __name__ == "__main__":
    # runs for past point and past line (500 each)

    # dataDirectory, savePathDirectory
    dataDirectory = "/mnt/talapasData/data/pastLine/1000/Line/"
    savePathDirectory = "/mnt/talapasData/data/pastLine/1000/Line/"
    lookAtNum = 500

    # line
    # iterates through all the files in dataDirectory and writes statsfile in SavePathDirectory
    calculateStatistics(dataDirectory, savePathDirectory, lookAtNum=lookAtNum, measurement='line')

    # point
    calculateStatistics(dataDirectory, savePathDirectory, lookAtNum=lookAtNum, measurement='point')
