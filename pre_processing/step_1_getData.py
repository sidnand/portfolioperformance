import os
import yfinance as yf

scriptDir = os.path.dirname(__file__)
relWritePath = "../data/new/orig"

sp_sectors = {
    "T_BILL_3_MO" : "%5EIRX",
    "SP_FINANCE" : "%5ESP500-40",
    "SP_ENEGY" : "%5EGSPE",
    "SP_MATERIALS" : "%5ESP500-15",
    "SP_CONSUM_DIS" : "%5ESP500-25",
    "SP_CONSUM_STAPLE" : "%5ESP500-30",
    "SP_HEALTH" : "%5ESP500-35",
    "SP_UTIL" : "%5ESP500-55",
    "SP_500" : "%5EGSPC",
    "SP_INFO_TECH" : "%5ESP500-45",
    "SP_TELE_COMM" : "%5ESP500-50"
}

industries = {}

def download(symbols, interval, start, end, path, folderName):
    outdir = os.path.join(scriptDir, path) + "/" + folderName

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for key, val in symbols.items():
        data = yf.download(val, group_by="Ticker", interval=interval, start=start, end=end)
        data.index = data.index.strftime("%Y%m%d")
        data.index = data.index.astype(int)
        data.to_csv(outdir + "/" + key + ".csv")

def get_data():
    # download sp500 sector data
    download(sp_sectors, "1mo", "1999-01-01", "2021-01-01", relWritePath, "sp_sector")