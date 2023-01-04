import os
import numpy as np

from util import readData

scriptDir = os.path.dirname(__file__)
relReadPath = "../data/new/pre_processed"
relWritePath = "../data/new/processed"

def processData(sector):
    indir = os.path.join(scriptDir, relReadPath)
    outdir = os.path.join(scriptDir, relWritePath)

    read = readData(indir + "/" + sector + ".csv", sepHeader=True)
    data = read[0]
    header = read[1]

    tBill = data[1:, 1]
    indices = data[1:, 2:]

    # compute difference between each indices and tBill
    diff = indices - tBill[:, np.newaxis]

    data = np.column_stack((tBill, diff))

    # create new file and save data
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # convert header[1:] to string comma separated
    header = ",".join(header[1:])

    np.savetxt(outdir + "/" + sector + ".csv", data, delimiter=',', fmt='%s', header=header, comments='')