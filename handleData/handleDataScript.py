from step1GetData import download
from step2CleanData import cleanData
from step3CombineData import combineData
from step4ProcessData import processData

sp_sectors = {
    "T_BILL_3_MO": "%5EIRX",
    "SP_FINANCE": "%5ESP500-40",
    "SP_ENEGY": "%5EGSPE",
    "SP_MATERIALS": "%5ESP500-15",
    "SP_CONSUM_DIS": "%5ESP500-25",
    "SP_CONSUM_STAPLE": "%5ESP500-30",
    "SP_HEALTH": "%5ESP500-35",
    "SP_UTIL": "%5ESP500-55",
    "SP_500": "%5EGSPC",
    "SP_INFO_TECH": "%5ESP500-45",
    "SP_TELE_COMM": "%5ESP500-50"
}

industries = {
    
}

download(sp_sectors, "1mo", "1999-01-01", "2021-01-01", "sp_sector")
cleanData("sp_sector")
combineData("sp_sector")
processData("sp_sector")