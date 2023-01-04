import os
import numpy as np

from util import *

DATE = 0
OPEN = 1
ADJ_CLOSE = 5

relReadPath = "../data/new/orig"
relWritePath = "../data/new/clean"


def cleanData(sector, readPath=relReadPath, writePath=relWritePath):
    scriptDir = os.path.dirname(__file__)

    indir = os.path.join(scriptDir, readPath, sector)
    outdir = os.path.join(scriptDir, writePath, sector)

    files = os.listdir(indir)
    # process each file
    for file in files:
        # read data
        data = readData(indir + "/" + file)

        if file[:6] == "T_BILL":
            data = removeUnnecessaryData(data, True)
        else:
            data = removeUnnecessaryData(data)
        
        # create new file and save data
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        np.savetxt(outdir + "/" + file, data, delimiter=',', fmt='%s', header="date,change", comments='')


def removeUnnecessaryData(data, tBill=False):
    if (tBill):
        data[:, ADJ_CLOSE] = data[:, ADJ_CLOSE] / 100
        newData = np.column_stack((data[:, DATE], data[:, ADJ_CLOSE]))
        return newData
    else:
        newData = percentageChange(data[:, OPEN], data[:, ADJ_CLOSE])
        newData = np.column_stack((data[:, DATE], newData))
        return newData
