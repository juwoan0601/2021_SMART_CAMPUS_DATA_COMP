TRAIN_DATASET_PATH          = "D:/POSTECH/교내활동/2021 스마트캠퍼스데이터경진대회/data/FINAL.csv"
TEST_DATASET_PATH           = "D:/POSTECH/교내활동/2021 스마트캠퍼스데이터경진대회/data/SUBMISSION.csv"

TARGET_COLUMN_NAME = 'HeadCount'

ALL_COULMNS = [
    'Date', 
    'Year', 
    'Month', 
    'Day', 
    'Hour', 
    'Week', 
    'Minute', 
    'HeadCount', 
    'Kcal', 
    'Protein', 
    'pork', 
    'beef', 
    'chicken', 
    'duck', 
    'processed meat', 
    'molluscs', 
    'fish', 
    'egg', 
    'dairy', 
    'tuna', 
    'shrimp', 
    'special', 
    'temperature(C)', 
    'precipitation(mm)', 
    'wind dir(deg)', 
    'wind speed(m/s)', 
    'spot atmospheric pressure(hPa)', 
    'sea-level pressure(hPa)', 
    'humidity(%)', 
    'sun radiation(MJ/m2)', 
    'bright sunshine(Sec)', 
    'fine dust concentration(ug/m3)', 
    'Cold wave warning', 
    'Heat wave warning', 
    'exam', 
    'class registration', 
    'make-up-class', 
    'vacation', 
    'Week N'
]

TEXT_COLUMNS = [
    'Date',
    'Week'
]

DATE_COLUMNS = [
    'Year', 
    'Month', 
    'Day', 
    'Hour',  
    'Minute',
    'Week N'
]

MENU_COLUMNS = [
    'pork', 
    'beef', 
    'chicken', 
    'duck', 
    'processed meat', 
    'molluscs', 
    'fish', 
    'egg', 
    'dairy', 
    'tuna', 
    'shrimp', 
    'special', 
]

WEATHER_COLUMNS = [
    'temperature(C)', 
    'precipitation(mm)', 
    'wind dir(deg)', 
    'wind speed(m/s)', 
    'spot atmospheric pressure(hPa)', 
    'sea-level pressure(hPa)', 
    'humidity(%)', 
    'sun radiation(MJ/m2)', 
    'bright sunshine(Sec)', 
    'fine dust concentration(ug/m3)', 
    'Cold wave warning', 
    'Heat wave warning',
    'Typoon'
]

SCHEDULE_COLUMNS = [
    'exam', 
    'class registration', 
    'make-up-class', 
    'vacation',
]
