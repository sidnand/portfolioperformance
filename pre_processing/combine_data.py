import os
import numpy as np

scriptDir = os.path.dirname(__file__)
relReadPath = "../data/new/clean"
relWritePath = "../data/new/pre_processed"

def readData(filename):
    # read csv file
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return data

def combineData(sector):
    indir = os.path.join(scriptDir, relReadPath)
    outdir = os.path.join(scriptDir, relWritePath)

    files = os.listdir(indir + "/" + sector)

    header = ""
    date = np.array([])
    change = np.array([])

    # process each file
    for file in files:
        # read data
        data = readData(indir + "/" + sector + "/" + file)

        if len(date) == 0:
            date = data[:, 0]
            header = "date"

        if len(change) == 0:
            change = data[:, 1]
        else:
            change = np.column_stack((change, data[:, 1]))

        header += "," + file[:-4]

    # new np array with date as first column and change as the rest
    data = np.column_stack((date, change))
    # date is int
    data[:, 0] = data[:, 0].astype(int)

    # create new file and save data
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    np.savetxt(outdir + "/" + sector + ".csv", data, delimiter=',', fmt='%s', header=header, comments='')

combineData("sp_sector")