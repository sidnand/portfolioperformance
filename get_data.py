import yfinance as yf

def get_SPSector():
    T_BILL_3_MO = "%" + "5EIRX"
    SP_FINANCE = "%" + "5ESP500-40"
    SP_ENEGY = "%" + "5EGSPE"
    SP_MATERIALS = "%" + "5ESP500-15"
    SP_CONSUM_DIS = "%" + "5ESP500-25"
    SP_CONSUM_STAPLE = "%" + "5ESP500-30"
    SP_HEALTH = "%" + "5ESP500-35"
    SP_UTIL = "%" + "5ESP500-55"
    SP_500 = "%" + "5EGSPC"
    SP_INFO_TECH = "%" + "5ESP500-45"
    SP_TELE_COMM = "%" + "5ESP500-50"

    SYMBOLS = [T_BILL_3_MO, SP_FINANCE, SP_ENEGY, SP_MATERIALS, SP_CONSUM_DIS, SP_CONSUM_STAPLE, SP_HEALTH, SP_UTIL, SP_500, SP_INFO_TECH, SP_TELE_COMM]

    return SYMBOLS

def download(symbols, interval, start, end):
    for ticker in symbols:
        data = yf.download(ticker, group_by="Ticker", interval=interval, start=start, end=end)
        data.to_csv(f'ticker_{ticker}.csv')  # ticker_AAPL.csv for example

# download sp500 sector data
symbols = get_SPSector()
download(symbols, "1mo", "2001-01-01", "2021-01-01")