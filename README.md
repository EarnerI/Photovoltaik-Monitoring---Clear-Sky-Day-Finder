# Clear Sky Day Filter for Photovoltaic Monitoring Data
A Clear Sky Day filter, that takes monitoring data of PV systems as power over time and returns only the data of clear sky days.
Usable for Module-, String- and Inverter data as well as for DC and AC power recordings.

<img width="375" height="287" alt="ClearSkyDay-GraphicalAbstract" src="https://github.com/user-attachments/assets/0156f08b-f373-40a0-b150-21ce0f698af8" /><center />  
  
Figure 1: The Clear Sky filter takes several days of monitoring data and returns only days, classified as clear sky days.

------------------------------------------------------------------------------------------
## Download via pip

pip install "git+https://github.com/EarnerI/Photovoltaik-Monitoring---Clear-Sky-Day-Finder.git"

## Code Example
from clearskydayfinder import get_clearskydays  
import polars as pl  
  
df = pl.read_csv("Example_Data.csv")  
  
clearskyday_df = get_clearskydays(df, column_time="time", column_power="power")  

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



 
 

