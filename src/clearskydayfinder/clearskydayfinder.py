import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from importlib import resources

def load_example_data():
    with resources.files("clearskydayfinder.data").joinpath("Example_Data.csv").open("rb") as f:
        data = pl.read_csv(f)
        return data

def get_clearskydays(data, column_time: str = "time", column_power: str = "power", column_id = None,
                     comparison_intervall: str = "30d", prep_smooth_kernal: int = None, smooth_kernal: int = None,
                     percentil: float = 0.9,
                     first_last_limit: float = 0.1, show_first_last_value: bool = True,
                     min_number_of_datapoints: int = None, find_numberofpoints: bool = True,
                     hole_size_threshold: int = 100, show_max_hole_size: bool = True,
                     plot_raw_data: bool = True,
                     corr_threshold: float = 0.98, plot_corr_results: bool = True,
                     max_dist: int = 40, n_max_exceeds: int = 50, plot_taken_results: bool = True):
    """
    writen by Ernst Wittmann August, 6th, 2025
    publication: E. Wittmann, C. Buerhop-Lutz, S. Bennett, V. Christlein, J. Hauch, C. J. Brabec, I. M. Peters,
                „PV Polaris – Automated PV system Orientation Prediction”, IEEE Photonics Journal, vol. 17,
                no. 3, 2025. DOI: 10.1109/JPHOT.2025.3568887
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Code Explanation:

    Short: Takes Monitoring Data of a PV system and filters clear sky days. Monitoring Data need contain at least time and power.

    The filter first removes all days with large gaps in the data or days not starting or ending with a power output of 0 W.
    The filter then generates a clear sky template. The process involves finding the maximum power output for each time of day
    over a one-month dataset, multiplying the result by a percentile (e.g., 0.9 for Germany), and smoothing it using median
    and mean sliding windows. Each day in the dataset is compared to the template using Euclidian distance and correlation analysis,
    and days having less of a user-defined number od distance exceeds and correlation threshold are labeled as clear sky days.

    ################################################################################################################################
    needed libaries: (these libaries were used when coded. Newer versions might work too.)
    import polars as pl || version 1.32.0  -- used as dataframe
    import matplotlib.pyplot as plt || version 3.10.5  -- used for helping plots
    from tqdm import tqdm || version 4.67.1 -- used for loading bars

    ################################################################################################################################
    Data Values:
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
    """
    clear_sky_frame = pl.DataFrame()
    # If no Id-column exists, then add one
    if column_id == None:
        column_id = "default_id"
        data = data.with_columns((pl.col(column_power) * 0).alias("default_id"))
    saving = data.clone()
    data = data[column_time, column_power, column_id]

    for module_data in tqdm(data.partition_by(column_id)):
        # -----------------------------------------------------------
        # SET DEFAULTS
        frequence = get_frequence(module_data, column_time) * 60
        if (min_number_of_datapoints == None):
            if (frequence <= 1):
                min_number_of_datapoints = 300
            elif (frequence > 1) & ((frequence <= 5)):
                min_number_of_datapoints = 100
            elif (frequence > 5) & ((frequence <= 10)):
                min_number_of_datapoints = 70
            elif (frequence > 500) & ((frequence <= 1000)):
                min_number_of_datapoints = 45
            else:
                min_number_of_datapoints = 20
            print(f"Rec_time: {frequence}s -> min_number_of_datapoints: {min_number_of_datapoints}")
        if (prep_smooth_kernal == None):
            if (frequence <= 1):
                prep_smooth_kernal = 10
            elif (frequence > 500) & ((frequence <= 1000)):
                prep_smooth_kernal = 5
            else:
                prep_smooth_kernal = 2
            print(f"Rec_time: {frequence}s -> prep_smooth_kernal: {prep_smooth_kernal}")
        if (smooth_kernal == None):
            if (frequence <= 1):
                smooth_kernal = 60
            elif (frequence > 1) & ((frequence <= 5)):
                smooth_kernal = 30
            elif (frequence > 5) & ((frequence <= 10)):
                smooth_kernal = 15
            elif (frequence > 500) & ((frequence <= 1000)):
                smooth_kernal = 10
            else:
                smooth_kernal = 2
            print(f"Rec_time: {frequence}s -> smooth_kernal: {smooth_kernal}")
        if (hole_size_threshold == None):
            if (frequence <= 1):
                hole_size_threshold = 60
            elif (frequence > 1) & ((frequence <= 5)):
                hole_size_threshold = 60
            elif (frequence > 5) & ((frequence <= 10)):
                hole_size_threshold = 40
            elif (frequence > 500) & ((frequence <= 1000)):
                hole_size_threshold = 30
            else:
                min_number_of_datapoints = 30
            print(f"Rec_time: {frequence}s -> hole_size_threshold: {hole_size_threshold}")
        if (n_max_exceeds == None):
            if (frequence <= 1):
                n_max_exceeds = 300
            elif (frequence > 1) & ((frequence <= 5)):
                n_max_exceeds = 100
            elif (frequence > 5) & ((frequence <= 10)):
                n_max_exceeds = 70
            elif (frequence > 500) & ((frequence <= 1000)):
                n_max_exceeds = 20
            else:
                n_max_exceeds = 5
            print(f"n_max_exceeds: {n_max_exceeds}")
        if (max_dist == None):
            max_dist = np.max(module_data[column_power].to_numpy()) * 0.3
            print(f"max_dist: {max_dist}")
        if find_numberofpoints:
            print("FIND MINIMAL NUMBER OF DATAPOINTS PER DAY:")
            _, counts = np.unique(module_data[column_time].dt.date(), return_counts=True)
            counts, bins = np.histogram(counts)
            counts = np.cumsum(counts)
            plt.stairs(counts[::-1], bins)
            plt.axvline(min_number_of_datapoints, color="red", label="Min number of points threshold.")
            plt.xlabel("number of points over one day", fontsize=20)
            plt.ylabel("number of days", fontsize=20)
            plt.tick_params(labelsize=18)
            plt.legend(fontsize=18)
            plt.show()
            print("")
            print("Only days that have more points then the threshold are taken into account!")
            print(f"Your recording time is estimated to {get_frequence(module_data, column_time) * 60}s")
            print(f"Thus, the maximal amount of datapoints per day is {1440 / get_frequence(module_data, column_time)}")
            print(f"You should put the threshold limit, so that you remove days with very low recordings, but keep the days with enough datapoints.")
            print(f"The current default value is set to {min_number_of_datapoints}.")
            print("(Note: to hide the plot and hint, set find_numberofpoints=False).")
            print("---------------------------------------------------------------------------")
        # ---------------------------------------------------------------
        for time_interval in module_data.sort(column_time).group_by_dynamic(
                column_time,  # The column to base the dynamic window on
                every=comparison_intervall,  # The interval of the windows
                closed="right"  # Determines if the interval includes the right boundary
        ).agg([pl.col(column_time).alias("m_time"),
               pl.col(column_power).alias("m_power"),
               pl.col(column_id).alias("m_id")]).iter_rows():
            month_data = pl.DataFrame({column_time: time_interval[1],
                                       column_power: time_interval[2],
                                       column_id: time_interval[3]})
            month_data = month_data.with_columns(pl.col(column_time).dt.date().alias("date"))

            # Calculate Clear Sky Template:
            cst = get_clearskytemplate(month_data, prep_smooth_kernal=prep_smooth_kernal, smooth_kernal=smooth_kernal,
                                       percentil=percentil, column_time=column_time, column_power=column_power)

            for count, day_interval in enumerate(month_data.group_by("date").agg([pl.col(column_time).alias("m_time"),
                                                                 pl.col(column_power).alias("m_power"),
                                                                 pl.col(column_id).alias("m_id")]).iter_rows()):
                day_data = pl.DataFrame({column_time: day_interval[1],
                                         column_power: day_interval[2],
                                         column_id: day_interval[3]})

                if len(day_data) > min_number_of_datapoints:  # Take day if enough datapoints
                    day_data = add_daytime(day_data, column_time=column_time).sort("day_time")
                    first_value_check = day_data[column_power].to_numpy()[0] <= first_last_limit
                    last_value_check = day_data[column_power].to_numpy()[-1] <= first_last_limit
                    if show_first_last_value:
                        print("FIND FIRST AND LAST LIMIT:")
                        plt.plot(day_data["day_time"], day_data[column_power], label="Data")
                        plt.axhline(first_last_limit, color="red", label="first_last_limit")
                        plt.xlabel("day time / s", fontsize=20)
                        plt.ylabel(column_power, fontsize=20)
                        plt.tick_params(labelsize=18)
                        plt.legend(fontsize=18)
                        plt.show()
                        print(f"First Value: {first_value_check} || Last Value: {last_value_check}")
                        if count == 0:
                            print(""" 
                        Both conditions must be True. If not, adjust the first_last_limit parameter accordingly (current value: {first_last_limit}).
                        
                        This check ensures that each day’s data begins and ends below the specified limit. 
                        The underlying assumption is that no power is generated during nighttime, so a 
                        complete and properly recorded day should start and end at approximately 0 W.
                        If a day does not start or end near 0 W, it typically indicates that the recording 
                        began or stopped during daytime hours, resulting in incomplete daily data.
                        (Note: To disable this hint, set show_first_last_value = False).""")
                        print("-------------------------------------------------------------------")
                    if first_value_check & last_value_check:  # Take day if starts/ends with night power generation (0W)
                        check_hole, hole_size = _bad_holes_check(day_data, hole_size_threshold=hole_size_threshold,
                                                                 column_time=column_time)
                        if show_max_hole_size:
                            print(f" HOLE_SIZE_THREESHOLD: {hole_size_threshold} / {hole_size}")
                            if count==0: print("""
                            The hole_size_threshold ({hole_size_threshold}) must be greater than the maximum 
                            detected gap ({hole_size}).
                            Data gaps (“holes”) can occur when no measurements are recorded 
                            for a period of time within a day. The hole_size_threshold defines the maximum allowed 
                            time (in minutes) between two consecutive data points. This limit must not be exceeded.
                            If a gap exceeds this threshold, it indicates missing data beyond the acceptable range.
                            
                            To disable this validation, set hole_size_threshold to a value greater than 1440 minutes 
                            (longer than a full day).
                            If set to None, a default value is automatically selected based on the recording frequency.
                            (Note: To hide this hint, set show_max_hole_size = False).""")
                        if check_hole:  # Take day if no lagging data.
                            if plot_raw_data:
                                print("----------- PLOT RAW DATA ------------------")
                                plt.plot(day_data["day_time"], day_data[column_power], label="Recorded Power")
                                plt.plot(cst["day_time"], cst["tmp_" + column_power], color="red", label="Power Template")
                                plt.xlabel("t/s", fontsize=20)
                                plt.ylabel(column_power, fontsize=20)
                                plt.tick_params(labelsize=18)
                                plt.legend(fontsize=18)
                                plt.grid()
                                plt.show()
                                print(f"Module ID: {module_data[column_id].to_numpy()[0]}" +
                                      f" || DAY: {day_interval[0]}")
                                print(f"perp_smooth_kernal: {prep_smooth_kernal}")
                                print(f"smooth_kernal: {smooth_kernal}")
                                if count==0:
                                    print(f"""
                                This plot shows the comparison between the recorded power data 
                                and the derived power template.The template is computed over the 
                                interval comparison_interval = {comparison_intervall}. 
                                It is first smoothed using prep_smooth_kernel = {prep_smooth_kernal}, 
                                followed by a second smoothing step with smooth_kernal = {smooth_kernal}.

                                Carefully evaluate the degree of smoothing:
                                Excessive smoothing can suppress important features, potentially resulting in an 
                                unusable or overly flattened template.
                                Insufficient smoothing may retain too much noise, leading to an unstable 
                                or noisy template.

                                Clear-sky days are identified as those that correlate well with the template. 
                                Therefore, the template should closely resemble a typical clear-sky power curve.

                                If the template quality is poor: 
                                The selected comparison_interval may be too short.
                                The smoothing parameters may need adjustment.

                                If the template’s peak power is systematically higher or lower than the recorded data, 
                                adjust the scaling factor percentile = {percentil}. This parameter scales the template 
                                and can account for effects such as atmospheric attenuation (e.g., dust).
                                Pertencile can have values between 0 to 1.
                                1: perfect situation.
                                0: no irradiance reaches the module due to soiling and air pollution.

                                (Note: To hide this hint, set plot_raw_data = False.)""")

                            comp_data = day_data.join_asof(cst, on="day_time", strategy="backward").drop_nulls()
                            corr = np.corrcoef(comp_data[column_power].to_numpy(),
                                               comp_data["tmp_" + column_power].to_numpy())[0, 1]

                            if plot_corr_results:
                                print("CORRELATION RESULT:")
                                plt.plot(comp_data["day_time"], comp_data[column_power], color="blue",
                                         label="Real Data")
                                plt.plot(comp_data["day_time"], comp_data["tmp_" + column_power], color="red",
                                         label="Template")
                                plot_data = comp_data.with_columns((pl.col(column_power) - pl.col("tmp_" + column_power)).abs().alias("dif"))
                                plot_data = plot_data.with_columns(pl.when(pl.col("dif")>max_dist).then(1).otherwise(0).alias("dif"))
                                plot_data = plot_data.filter(pl.col("dif")==1)
                                plt.vlines(plot_data["day_time"],plot_data[column_power], plot_data["tmp_" + column_power], color="orange", label="Violations")
                                plt.xlabel("t/s", fontsize=20)
                                plt.ylabel(column_power, fontsize=20)
                                plt.tick_params(labelsize=18)
                                plt.legend(fontsize=18)
                                plt.grid()
                                plt.show()
                                print(f"Module ID: {module_data[column_id].to_numpy()[0]}" +
                                      f" || DAY: {day_interval[0]}")
                                print(f"CORRELATION: {corr_threshold}/{corr}")
                                print(f"Max_dist: {max_dist} and n_max_exceeds: {n_max_exceeds}  -> Violations: {len(comp_data.filter((pl.col(column_power) - pl.col('tmp_' + column_power)) < -max_dist))}")
                                if count == 0:
                                    print(f"""
                                    Only days with a correlation higher than corr_threshold = {corr_threshold} are considered.
                                    The current day has a correlation of {corr}.
                                    
                                    !Additionally, a day is only accepted if the recorded power does not fall below the 
                                    template by more than max_dist = {max_dist} 
                                    for more than n_max_exceeds = {n_max_exceeds} data points.
                                    max_dist is in this case the power difference between template and recording.
                                    Depending on the input data it can be a value given in W, kW or if the data is normed a value between 0 to 1.
                                    n_max_exceeds is the number of violations, the number how of often the maximal allowed power difference is allowed.

                                    Number of exceedances (violations): {len(comp_data.filter((pl.col(column_power) - pl.col('tmp_' + column_power)) < -max_dist))}

                                    Interpretation:
                                    An exceedance occurs when the recorded power is  lower than the 
                                    template beyond the allowed tolerance (max_dist = {max_dist}). 
                                    Too many exceedances (n_max_exceeds: {n_max_exceeds}) indicate 
                                    that the day deviates too strongly from the expected clear-sky profile.
                                    
                                    In order do ignore e.g. soiling losses, you can reduce the pertencile. 
                                    Pertencile can have values between 0 to 1.
                                    1: perfect situation.
                                    0: no irradiance reaches the module due to soiling and air pollution.

                                    (Note: To hide the plot and these hints, set plot_corr_results = False).""")

                            if corr > corr_threshold:  # Take day if correlating with template
                                if _check_distance(comp_data, max_dist=max_dist, n_max_exceeds=n_max_exceeds,
                                                column_power=column_power):  # Take day if distance is accepted
                                    day_data = day_data.with_columns((pl.col(column_power) * 0 + corr).alias("Corr"))
                                    clear_sky_frame = pl.concat([clear_sky_frame, day_data])

                                    if plot_taken_results:
                                        print("-#-#-#-#-#- CLEAR SKY DAYS -#-#-#-#-#-")
                                    print(f"Module ID: {module_data[column_id].to_numpy()[0]}" +
                                          f" || DAY: {day_interval[0]}")
                                    print(f"DATAPOINTS: {len(day_data)}")
                                    print(f"CORRELATION: {corr}")
                                    print("Hole: ", hole_size)
                                    print(f"Violations: {len(comp_data.filter((pl.col(column_power) - pl.col('tmp_' + column_power)) < -max_dist))}")
                                    plt.figure(figsize=(4, 4))
                                    plt.plot(day_data["day_time"], day_data[column_power], color="blue",
                                             label="Real Data")
                                    plt.plot(comp_data["day_time"], comp_data["tmp_" + column_power], color="red",
                                             label="Template")
                                    plt.legend(fontsize=16)
                                    plt.xlabel("time / h", fontsize=18)
                                    plt.ylabel("P / W", fontsize=18)
                                    plt.grid()
                                    plt.tick_params(labelsize=16)
                                    plt.xlim(0, 1440)
                                    plt.tight_layout()
                                    #plt.savefig(f"corr{corr}.png", dpi=500)
                                    plt.show()
                                    print("(Note, to hide the plot, do plot_taken_results=False.)")
                                    print("-------------------------------------------------------------")
    clear_sky_frame = clear_sky_frame.join(saving, on= [column_time, column_power, column_id])
    return clear_sky_frame

def get_clearskytemplate(id_data, column_time: str="time", column_power: str="power", percentil: float=0.9, prep_smooth_kernal: int=10, smooth_kernal: int=60):
    """
    Takes a dataframe of monitoring data, including power over time, over several days and creates a maximum value curve. (Clear sky template)
    Afterwards the daily curve is multiplied by a percentil, to correct the daily curve. The reason for correction is due to e.g. air polution.

    see publication:
    I. M. Peters, S. Karthik, L. Haohui, T. Buonassisi and A. Nobre, “Urban Haze and Photovoltaics,” Energy & Environmental Science, 2018.

    :param id_data: a data_frame containing power over time, given in two columns.
    Important: Care, that the power is only given of ONE PV system at a time. If there are two PV systems in the same power_column
    this can and will lead to a wrongly calculated clear sky template!!
    :param column_time: the column name including the time values as [utc] in [ns]; (default: "time")
    :param column_power: the column name including the power values as [float32 or flout64]; (default: "power")
    :param prep_smooth_kernal: A sliding mean window, that can be used "before" the maximal values are determined for each time step.
                               -> Reduces noise of the raw days
    :param smooth_kernal: A sliding mean window, that can be used "after" the maximal values are determined for each time step.
                          -> smoothes the resulting maximal value curve.
    :param percentil: correction factor; (in Germany probably 0.9); (default: 0.9)

    :return: a polars dataframe, including the power over time of a maximal power curve over all included days given in column "tmp_{column_power}".
    """
    cst = add_daytime(id_data,column_time=column_time).sort("day_time")
    real_max_value = cst[column_power].max()
    if prep_smooth_kernal>1:
        cst = cst.with_columns(pl.col(column_power).rolling_mean(window_size=prep_smooth_kernal, center=True))
    cst = cst.group_by("day_time").agg(pl.col(column_power).max())
    cst = cst.with_columns(pl.col(column_power).alias(f"tmp_{column_power}"))
    if smooth_kernal>1:
        cst = cst.with_columns(pl.col(column_power).rolling_mean(window_size=smooth_kernal, center=True).alias(f"tmp_{column_power}"))
    smooth_max_value = cst[f"tmp_{column_power}"].max()
    cst = cst.with_columns(pl.col(f"tmp_{column_power}")/smooth_max_value*real_max_value)
    cst = cst.with_columns(pl.col(f"tmp_{column_power}") * percentil)
    return cst

def add_daytime(day_data,column_time: str="time"):
    """
    Takes a data frame including time and adds a new column called "day_time" including the day time in minuts.
    :param day_data: polars dataframe including a time column
    :param column_time: the column name including the time values as [utc] in [ns]; (default: "time")
    :return: a polars data frame including a new column called "day_time" including the time of the day in minutes.
    """
    if "day_time" not in day_data.columns:
        return day_data.with_columns((pl.col(column_time).dt.hour().cast(pl.Float32) * 60
                                  + pl.col(column_time).dt.minute().cast(pl.Float32)
                                  + pl.col(column_time).dt.second().cast(pl.Float32) / 60).alias("day_time"))
    else:
        return day_data

def _bad_holes_check(id_data, hole_size_threshold: int=20, column_time: str="time"):
    """
    Takes a data set including time. Checks, if the time between two timesteps is less than a given hole_size_threshold in minutes.
    :param day_data: polars frame with time
    :param hole_size_threshold: maximal allowed time in minutes between two timesteps.
    :param column_time: the column name including the time values as [utc] in [ns]; (default: "time")
    :return:
    """
    array = add_daytime(id_data, column_time=column_time).sort("day_time")["day_time"].to_numpy()
    if len(array) > 4:
        dif = array[1:] - array[:-1]
        max_hole = np.nanmax(dif)
        if max_hole < hole_size_threshold:
            check = True
        else:
            check = False
    else:
        max_hole = None
        check = False
    return check, max_hole

def _check_distance(comp_data, max_dist: int=10, n_max_exceeds: int=50, column_power: str="power"):
    """Calculates the Euclidian Distances for each timestep between recoded power and clear sky template."""
    difs = np.abs(comp_data[column_power].to_numpy() - comp_data["tmp_"+column_power].to_numpy())
    difs = difs[difs>max_dist]
    if len(difs)>n_max_exceeds: return False
    else: return True

def get_frequence(id_data, column_time: str="time"):
    """
    Takes a polars dataframe, and calculates the median time in minutes between two timesteps.
    :param id_data: data frame as polars including time
    :param column_time: the column name including the time values as [utc] in [ns]; (default: "time")
    :return: median time in minutes between two time steps.
    """
    id_data = add_daytime(id_data, column_time=column_time).sort(column_time)
    return np.nanmedian((id_data["day_time"].to_numpy()[1:]-id_data["day_time"].to_numpy()[:-1]))
