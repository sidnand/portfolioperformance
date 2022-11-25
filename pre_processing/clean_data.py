import os
import numpy as np

DATE = 0
OPEN = 1
ADJ_CLOSE = 5

scriptDir = os.path.dirname(__file__)
relReadPath = "../data/new/orig"
relWritePath = "../data/new/clean"

def readData(filename):
    # read csv file
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    # date is int
    return data

def percentageChange(open, close):
    # return percentage change as a percentage
    return (close - open) / open

def processData(data, tBill=False):
    if (tBill):
        data[:, ADJ_CLOSE] = data[:, ADJ_CLOSE] / 100
        newData = np.column_stack((data[:, DATE], data[:, ADJ_CLOSE]))
        return newData
    else:
        newData = percentageChange(data[:, OPEN], data[:, ADJ_CLOSE])
        newData = np.column_stack((data[:, DATE], newData))
        return newData

def processSector(readPath, sector, writePath):
    indir = os.path.join(scriptDir, readPath, sector)
    outdir = os.path.join(scriptDir, writePath, sector)

    files = os.listdir(indir)
    # process each file
    for file in files:
        # read data
        data = readData(indir + "/" + file)

        if file[:6] == "T_BILL":
            data = processData(data, True)
        else:
            data = processData(data)
        
        # create new file and save data
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        np.savetxt(outdir + "/" + file, data, delimiter=',', fmt='%s', header="date,change", comments='')

def clean_data():
    processSector(relReadPath, "sp_sector", relWritePath)