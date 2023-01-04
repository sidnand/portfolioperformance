import os
import yfinance as yf

relWritePath = "../data/new/orig"

def download(symbols, interval, start, end, folderName, relWritePath=relWritePath):
    scriptDir = os.path.dirname(__file__)
    outdir = os.path.join(scriptDir, relWritePath) + "/" + folderName

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for key, val in symbols.items():
        data = yf.download(val, group_by="Ticker", interval=interval, start=start, end=end)
        data.index = data.index.strftime("%Y%m%d")
        data.index = data.index.astype(int)
        data.to_csv(outdir + "/" + key + ".csv")