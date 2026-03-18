# Clear Sky Day Filter for Photovoltaic Monitoring Data
A Clear Sky Day filter, that takes monitoring data of PV systems as power over time and returns only the data of clear sky days.
Usable for Module-, String- and Inverter data as well as for DC and AC power recordings.

<img width="375" height="287" alt="ClearSkyDay-GraphicalAbstract" src="https://github.com/user-attachments/assets/0156f08b-f373-40a0-b150-21ce0f698af8" /><center />  
  
Figure 1: The Clear Sky filter takes several days of monitoring data and returns only days, classified as clear sky days.

------------------------------------------------------------------------------------------
## Download via pip

pip install "git+https://github.com/EarnerI/Photovoltaik-Monitoring---Clear-Sky-Day-Finder.git"

## Short Code Example
```python
from clearskydayfinder import get_clearskydays, load_example_data

# Frist we load the example data frame including power over time of two PV modules each with its own module_id:
data = load_example_data()

# The time column has to include date time:
data = data.with_columns(pl.col("time").str.to_datetime().alias("time"))

# To get a data_frame only including the clear sky days do:
clearskyday_df = get_clearskydays(data, column_time="time", column_power="power", column_id = "module_id")

# (An id is not nessecarly needed!)
```

## Explanations:

get_clearskydays(data, column_time: str = "time", column_power: str = "power", column_id = None, comparison_intervall: str = "30d", prep_smooth_kernal: int = None, smooth_kernal: int = None, percentil: float = 0.9, first_last_limit: float = 0.1, show_first_last_value: bool = True, min_number_of_datapoints: int = None, find_numberofpoints: bool = True, hole_size_threshold: int = 100, show_max_hole_size: bool = True, plot_raw_data: bool = True, corr_threshold: float = 0.98, plot_corr_results: bool = True, max_dist: int = 40, n_max_exceeds: int = 50, plot_taken_results: bool = True)

:param data: polar frame, including time and power
:param column_time: the column name including the time values as [utc] in [ns]; (default: "time")
:param column_power: the column name including the power values as [float32 or flout64]; (default: "power")
:param column_id: the column name including the ids:
If there are several PV systems, there are two ways the systems are stored in a dataframe.
1. each power recording of a PV system has its own column, like: {columns: time, system1, system2,...}. In this case
    column_id has to be "None". In this case for each system, the code has to be run and the input of "column_power"
    has to be updated.
2. there is one power column and an id column. In this case, put the name of the id column.
(default: module_id)
-----------------------------------------------------------------------------------------------------------------------------------------
Clearsky-Template:
Takes a dataframe of monitoring data, including power over time, over several days and creates a maximum value curve. (Clear sky template)
Afterwards the daily curve is multiplied by a percentil, to correct the daily curve. The reason for correction is due to e.g. air polution.
:param comparison_intervall: the intervall of taken days for the templates. (default: 30d)
:param prep_smooth_kernal: A sliding mean window, that can be used "before" the maximal values are determined for each time step.
                           -> Reduces noise of the raw days
:param smooth_kernal: A sliding mean window, that can be used "after" the maximal values are determined for each time step.
                      -> smoothes the resulting maximal value curve.
:param percentil: correction factor; (in Germany probably 0.9); (default: 0.9)
------------------------------------------------------------------------------------------------------------------------------------------
General Day filtering:
:param min_number_of_datapoints: the minimal amount of datapoints ONE DAY has to include in order to be recognised as possible clear sky day.
                                 This value strongly depends on the recording frequence.
                                 E.g.:  A day has 24h, if the recording speed is one datapoint per 30min, this should be 48 datapoints.
                                        If one only accepts full day recordings, min_number_of datapoints should be 48.
                                 If set to "None" the value is filled automatically by a default value.However the default values are not the best values.
                                 In order to find a good value, one can set "find_numberofpoints" to "True".
                                 If you dont want to have this check, than just set the value very high.

:param find_numberofpoints: Shows a plot, that presents how many days have how many datapoints and the current min_number_of_datapoints threhold. (default: True)
:param first_last_limit: The days are checked, if they start and end with a value lower than this limit.
                         The idea behind it is, that in the night, no power is generated, therefore a full recorded day, should start and end with 0 W.
                         If a day recodring does not start/end wih 0 W, this means typical, that the recording started/ended during the day.
                         One can check the first and last day values, by setting "show_first_last_value".
                         If you dont want to have this check, than just set the value very high.
                         (default: 0.1)
:param show_first_last_value: prints the first and last value of each checked day. If one of the values is higher than first_last_limit the day is filtered out.(default: False)
:param hole_size_threshold: It is possible, that during a day there is no data recorded -> leading to a data hole.
                            Within hole_size_threshold the time between two recording steps can be given in minutes, that should NOT be exceeded.
                            If this check should not be considered, just set the value higher than 1440.
                            If set to None, a default value depending on the frequendy is used.
:param show_max_hole_size: prints the maximal hole_size for each day.
:param plot_raw_data: plots each day, that fullfills the checks given by the limits above.
-------------------------------------------------------------------------------------------------------------------------------------------------
Correlation Comparison
:param corr_threshold:  Each day is compared with the given template. Comparison is done by the correlation.
                        If the correlation is higher than corr_threshold, the day is accepted as possible clear sky day.
                        Range: 0-1 -- 0: Every day will be choosen as possible clear sky day.
                                      1: Only exactly correlating days will be accepted.
                        Good values are typically between 0.95 - 0.98 (good quantity to quality relation)
:param plot_corr_results: plots all days, that exceed the correlation threshold.
-----------------------------------------------------------------------------------------------------------------------------------------------
Euclidian Distance Comparison
:param max_dist: The maximal allowed power difference for a time step. If the Distance is exceeded for a time step, the step is counted.
                 This value depends on the P_MPP
:param n_max_exceeds: The maximal count of time steps exceeding the maximal allowed power difference.
                      This value depends on the frequency.
                      Default values are given depending on the frequency.

:param plot_taken_results: Plots the the taken days counted as clear sky days.
-----------------------------------------------------------------------------------------------------------------------------------------------
:return: Returns a dataframe including only the filtered clear sky days, given as polars dataframe.

## Parameter Fitting:
```python
from clearskydayfinder import get_clearskydays, load_example_data
import polars as pl
# Frist we load the example data frame including power over time of two PV modules each with its own module_id:
data = load_example_data()
# The time column has to include date time:
data = data.with_columns(pl.col("time").str.to_datetime().alias("time"))
```
The first parameter to fit is the "min_number_of_datapoints.
Running "get_clearskyday" will plot a bar plot of the number of data points over days if find_numberofpoints=True.
The plot is supposed to help finding good value for min_number_of_datapoints.
min_number_of_datapoints has default values based on the recording frequency, however manual adjustment is recommended.
```python
clearskyday_df = get_clearskydays(data, column_time = "time", column_power = "power", column_id = "module_id",
                     min_number_of_datapoints= None, find_numberofpoints: bool = True,
                    show_first_last_value = False,
                     show_max_hole_size= False,
                     plot_raw_data= False,
                     plot_corr_results = False,
                     plot_taken_results = False)  
```
<img width="487" height="384" alt="grafik" src="https://github.com/user-attachments/assets/53e4ee44-4056-4095-89f0-eef9c60918a5" />

Since the data is recorded over each minute (Recordingintervall: 52.99 sec), a full day should include about  1440 datapoints (24h * 60).
You can see in the plot, that the example data has about 200 days with only 300 datapoints per day, within about 400 points over days, the number of days is decreasing until 1000 points over a day is reached.
With setting min_number_of_datapoints all days with less datapoints over a day are removed and only days with more then min_number_of_datapoints are used. 


The next parameter is the first_last_value (dafault: 0.1).
The idea is, to remove days where the recording did not start and end at the beginnen and end of a day. 
The value first_last_limit tells where the start and end value has to lie among.
```python

# Since the data is interpolated, it will always start and end at 0W. Therfore we remove all values lower than 0.1W.
data = data.filter(pl.col("power")>0.1)

# Now we search only for days starting and ending below 50W:
clearskyday_df = get_clearskydays(data, column_time = "time", column_power = "power", column_id = "module_id",
                     min_number_of_datapoints= 600, find_numberofpoints: bool = True,
                     first_last_limit = 10, show_first_last_value = False,
                     show_max_hole_size= False,
                     plot_raw_data= False,
                     plot_corr_results = False,
                     plot_taken_results = False)  
```

<img width="1391" height="531" alt="grafik" src="https://github.com/user-attachments/assets/6d6532c0-47c1-4b45-8f54-dbf5623ac7c9" />


The next values that need to be adjusted are smoothing parameters prep_smooth_kernal and smooth_kernal for the template.




## Citation
If you use the code in terms of a publication, I would be greatfull, if you could cite the following paper:  
E. Wittmann, C. Buerhop-Lutz, S. Bennett, V. Christlein, J. Hauch, C. J. Brabec, I. M. Peters, „PV Polaris – Automated PV system Orientation Prediction”, IEEE Photonics Journal, vol. 17, no. 3, 2025. DOI: 10.1109/JPHOT.2025.3568887

-----------------------------------------------------------------------------------------------


## Functionality:
First a clear sky template is created based on the work of Ian Marius et all. [1]. The process involves finding the maximum power output for each time of day over a one-month dataset, multiplying the result by a percentile (e.g., 0.9 for Germany), and smoothing it using median and mean sliding windows.
The clear sky filter than filters out days, that show lagging data. To filter days with lagging data, there are three checks:
1. Check if data over a day has enough datapoints. (e.g. within a recordingspeed of 1h, there should be 24 datapoints for a day)
2. Check if there are data holes.
3. Check if the power curve over a day starts and ends by low power. (day starts and ends at night and during night the power generation is typically low)
Afterwards the remaining days are compared with the template by Euclidian distance and correlation thresholds for similarity. If a day is similar (low distances and high correlation), the day is defined as clear sky day.



 
 

