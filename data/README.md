This folder should contain all data downloads.

The data should be sourced from [Dukascopy Historical Data Feed](https://www.dukascopy.com/swiss/english/marketwatch/historical/).
The data should have the following format:
- Within a csv file, there should be the following columns: "date_gmt", "open", "high", "low", "close", "volume".
- Volume should be in units, not thousands or millions. 
- The date of each candle should be in GMT.

An Example of getting the data from the [Dukascopy Historical Data Feed](https://www.dukascopy.com/swiss/english/marketwatch/historical/):
![image](https://github.com/user-attachments/assets/6b96d1e6-a496-4933-9305-a38ab1589f9e)


The folders should be structured as the following: 
```
FILENAME = {FromDate(DD.MM.YYYYTHH:mm)-ToIncludingDate(DD.MM.YYYYTHH:mm)} // where the first date is the gmt date of the first candle, and the second date is the gmt date and time of the last candlestick.
FOLDER = {InstrumentType}/{Instrument}/{Granularity}/{BID/ASK}/FILENAME.csv
```
