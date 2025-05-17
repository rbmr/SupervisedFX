This folder should contain all data downloads.
Python files should go in common/

The data should be sourced from [Dukascopy Historical Data Feed](https://www.dukascopy.com/swiss/english/marketwatch/historical/).
The data should have the following format:
- Within a csv file, there should be the following columns: "date_gmt", "open", "high", "low", "close", "volume".
- Volume should be in units, not thousands or millions. 
- The date of each candle should be in GMT.

An Example of getting the data from the [Dukascopy Historical Data Feed](https://www.dukascopy.com/swiss/english/marketwatch/historical/):
![image](https://github.com/user-attachments/assets/6b96d1e6-a496-4933-9305-a38ab1589f9e)


The files should be saved in the following manner: 
```
forex/{Instrument}/{Granularity}/{BID/ASK}/{Start}-{End}.csv
```
Where:
- `Instrument` is written like `EURUSD`, `AUDUSD`, `GBPUSD`, etc
- `Granularity` is written like `1M`, `15M`, `1H`, etc.
- Both `Start` and `End` are formatted like: `DD.MM.YYYYTHH.mm` 
- `Start` is the time of the first candle in the dataset, and `End` is the time of the last candle.

Example:

```
forex/EURUSD/15_M/BID/10.05.2022T00.00-09.05.2025T20.45.csv
```