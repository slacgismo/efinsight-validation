import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __():
    import pandas as pd
    import marimo as mo
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime, timedelta
    return datetime, mo, np, os, pd, plt, timedelta


@app.cell
def __(mo):
    mo.md(
        r"""
        # Files Needed
        - sce_ami.csv.gz (SCE AMI data) for *sce_ami_data*
        - NHEC AMI files (the code reads the csv files in a specific folder) for *folder*
            - ElectricMeterReadings_INTERVALCIS_EXPORT_01-01-21-0000_07-31-21-0000.csv
            - ElectricMeterReadings_INTERVALCIS_EXPORT_01-01-22-0000_08-18-22-0000.csv
            - ElectricMeterReadings_INTERVALCIS_EXPORT_08-01-20-0000_12-31-20-0000.csv
            - ElectricMeterReadings_INTERVALCIS_EXPORT_08-01-21-0000_12-31-21-0000.csv
        - Organized AMI Data: sce_ami_edited.csv (too big for Github, needs to be created) and nhec_ami_edited.csv; these files need to be imported into the Loadshape pipeline
        - Loadshape.csv from the Loadshape pipeline for *kmeans_file*
        - resstock_metadata.csv and comstock_metadata.csv to find the number of buildings as part of the data normalization for *resstock_bldgs* and *comstock_bldgs*
        - Resstock and Comstock kW/sf data for all building types (each building type has a separate variable) under GISMo Forecast Data to run the graphs
        - CA-Orange.csv.zip for Orange County housing square footage data for *orange_cty*
        - Loadshape files for *nhec_ami_kmeans_2* and *sce_ami_kmeans_2s*
        - Any reference to k-means is refering to the output from the Loadshape pipeline, since this pipeline uses k-means clustering to sort load shapes
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""# SCE AMI Data""")
    return


@app.cell
def __(mo):
    mo.md(
        """
        - SCE AMI data comes from Orange County, California [Winchester]
            - This data contains commercial data (most likely) for 973 different buildings
        - This data will be organized by the OpenFIDO loadshape pipeline (https://github.com/openfido/loadshape), which requires an ID column, data column, timezone column, and a datetime column.
            - This pipeline groups houses with similar loadshapes. When using residential data, the group with the biggest count was assumed to be the group containing residential data
            - For UTC, since the offset is already included in the timestamp column, the timezone column has 0s.
            - For local time, daylight savings is recorded with a 1, and 0 refers to standard time.
        - Installing the loadshape pipeline on a Mac:
            - Activate your environment and changed the directory to the folder that has your config.csv and data file.
                - If you do not have a config.csv file, loadshape will create one in the selected folder during installation
            - Activate Arras Energy through Docker and use the command "docker run -itv $PWD:/app lfenergy/arras:develop bash"
                - This step can be optional in some cases
            - Install the pipeline with "openfido install loadshape"
            - In order for the pipeline to read your file, input "export OPENFIDO_INPUT=/app" and "export OPENFIDO_OUTPUT=/app"
            - In the config.csv file, input columns using their column names. Input your data file using its file name
                - Make sure to give "OUTPUT_PNG" a file name so it will output, it allows the user determine the residential group
                - The rest can stay the same, unless your datetime format is different
            - Group count is up to the user, but this Marimo document offers code to find the best group count for this pipeline
            - Run the pipeline using "openfido run loadshape"
            - Outputs
                - groups.csv: tells user the group each ID was assigned to
                - loadshapes.csv: tells the user the means for each seasonal, daytype, hour combination (192 combos) for each group
                - output.png: graphs all the means in loadshapes.csv for each group
        """
    )
    return


@app.cell
def __(pd):
    # Code to organize SCE AMI data for k-means

    # Read in data
    sce_ami_data = pd.read_csv("../sce_ami/sce_ami.csv.gz")

    # Organizing Timestamp data for both UTC and local timezones
    sce_ami_data['timestamp_utc'] = pd.DatetimeIndex(sce_ami_data['record_date']) + pd.to_timedelta(sce_ami_data['hour_id'].values - 1 \
                                                                                                    + sce_ami_data['utc_offset'], \
                                                                                                    unit = 'h')
    sce_ami_data['timestamp_local'] = pd.DatetimeIndex(sce_ami_data['record_date']) + pd.to_timedelta(sce_ami_data['hour_id'].values \
                                                                                                      - 1, unit = 'h')

    # Create daylight savings column for k-means
    sce_ami_data['is_dst'] = sce_ami_data['utc_offset'].apply(lambda x: 1 if x == 7 else 0)

    # Setting index columns, organizing, and adding timezone columns for k-means
    sce_ami_idx = sce_ami_data.set_index(['circuit_name', 'customer_id','timestamp_local'])
    sce_ami_drp = sce_ami_idx.drop(['record_id', 'record_date', 'hour_id', 'utc_offset', 'energy_unit'], axis=1)
    sce_ami = sce_ami_drp.sort_index(level=['circuit_name', 'customer_id'])
    sce_ami["timezone_utc"] = 0

    #Save Data
    #sce_ami.to_csv("../sce_ami/sce_ami_edited.csv", index=True) # Need to run this block first, the edited SCE AMI datafile is too large for GitHub
    return sce_ami, sce_ami_data, sce_ami_drp, sce_ami_idx


@app.cell
def __(sce_ami):
    sce_ami
    return


@app.cell
def __(plt, sce_ami):
    # Plotting AMI January data
    sce_ami_win = sce_ami.loc[sce_ami.index.get_level_values('timestamp_local').month == 1]
    sce_ami_w_idx = sce_ami_win.reset_index(level=['customer_id', 'circuit_name'])
    sce_ami_hour = sce_ami_w_idx.drop(['customer_id', 'circuit_name', 'is_dst', 'timestamp_utc', 'timezone_utc'], axis=1)
    sce_ami_hour['hour'] = sce_ami_hour.index.get_level_values('timestamp_local').hour
    sce_ami_jan = sce_ami_hour.groupby('hour')['energy_value'].mean()

    sce_ami_jan.plot(x=sce_ami_jan.index, y=sce_ami_jan.values, kind='line', linestyle='-', color='b', figsize=(20, 10))
    plt.xticks(range(min(sce_ami_jan.index), max(sce_ami_jan.index)+1, 2), fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Average Day in Jan from SCE AMI Data (Raw Data)', fontsize=20)
    plt.xlabel('Hours', fontsize=20)
    plt.ylabel('Mean Power for an Avg House (kW)', fontsize=20)
    plt.grid(True)
    plt.gca()
    return sce_ami_hour, sce_ami_jan, sce_ami_w_idx, sce_ami_win


@app.cell
def __(mo):
    mo.md(r"""# NHEC AMI Data""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        - NHEC AMI data comes from New Hampshire from the PostRoad-Energy Github Repo (https://github.com/postroad-energy/simulation/tree/develop-add-ami/nhec/ami)
        - Only local time was included in the organized dataset
        """
    )
    return


@app.cell
def __(datetime, os, pd):
    folder = "../NHEC_Files"
    data_list = []

    # Organizing AMI Data in a DataFrame with meter ids and timestamps as indexes; power is in kW
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.endswith('.csv'):
            data = pd.read_csv(file_path,
                                   index_col=['AMI Meter ID','Start Date'],
                                   usecols=['AMI Meter ID','Start Date','Kwh'],
                                   low_memory=False,
                                   parse_dates = ['Start Date'],
                                   )
            data.index.names=['meter_id','timestamp']
            data.columns=['power']
            data_list.append(data)
    data = pd.concat(data_list)
    nhec_ami = data.groupby([pd.Grouper(level='meter_id'),pd.Grouper(level='timestamp',freq='1h')]).agg({'power':'sum'}).sort_index()

    timestamps = nhec_ami.index.get_level_values('timestamp')
    utc_offset = []

    def add_utc(time):
        if (datetime(2020, 3, 8) <= time < datetime(2020, 11, 1)) or \
           (datetime(2021, 3, 14) <= time < datetime(2021, 11, 7)) or \
           (datetime(2022, 3, 13) <= time < datetime(2022, 11, 6)):
            return '4' 
        else:
            return '5'

    for _times in timestamps:
        utc = add_utc(_times)
        utc_offset.append(utc)

    dst_results = []

    def add_dst(time):
        if (datetime(2020, 3, 8) <= time < datetime(2020, 11, 1)) or \
           (datetime(2021, 3, 14) <= time < datetime(2021, 11, 7)) or \
           (datetime(2022, 3, 13) <= time < datetime(2022, 11, 6)):
            return '1'
        else:
            return '0'

    for _dates in timestamps:
        is_dst = add_dst(_dates)
        dst_results.append(is_dst)

    nhec_ami['is_dst'] = dst_results

    #nhec_ami.to_csv('../Loadshape_files/nhec_ami_edited.csv', index=True)
    return (
        add_dst,
        add_utc,
        data,
        data_list,
        dst_results,
        file_path,
        filename,
        folder,
        is_dst,
        nhec_ami,
        timestamps,
        utc,
        utc_offset,
    )


@app.cell
def __(nhec_ami):
    nhec_ami
    return


@app.cell
def __(nhec_ami, plt):
    # Plotting AMI January data
    nhec_ami_win = nhec_ami.loc[nhec_ami.index.get_level_values('timestamp').month == 1]
    nhec_ami_w_idx = nhec_ami_win.reset_index(level=['meter_id'])
    nhec_ami_hour = nhec_ami_w_idx.drop(['meter_id', 'is_dst'], axis=1)
    nhec_ami_hour['hour'] = nhec_ami_hour.index.get_level_values('timestamp').hour
    nhec_ami_jan = nhec_ami_hour.groupby('hour')['power'].mean()

    nhec_ami_jan.plot(x=nhec_ami_jan.index, y=nhec_ami_jan.values, kind='line', linestyle='-', color='b', figsize=(20, 10))
    plt.xticks(range(min(nhec_ami_jan.index), max(nhec_ami_jan.index)+1, 2), fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Average Day in Jan from NHEC AMI Data (Raw Data)', fontsize=20)
    plt.xlabel('Hours', fontsize=20)
    plt.ylabel('Mean Power for an Avg House (kW)', fontsize=20)
    plt.grid(True)
    plt.gca()
    return nhec_ami_hour, nhec_ami_jan, nhec_ami_w_idx, nhec_ami_win


@app.cell
def __(mo):
    mo.md(
        r"""
        # Finding the best Loadshape pipeline group count
        - Both the AMI data and Loadshape files are loaded in manually, then two functions are used to assign season and daytype to the data. Hour is also assigned, creating 3 new columns within the dataframe
        - Then, four dictionaries are created, one for each season, where each key has each daytype and hour combination, creating 48 keys
        - The standard deviation of the AMI data with the means from Loadshape are found. The means of the residential data and the overall data are also found. The following plots show the mean standard deviations for the SCE and NHEC AMI data when compared to their Loadshape files.
        - The standard deviation formula used is: $\sigma = \sqrt{\frac{\displaystyle\sum_{} (x_{i} - \mu)^{2}}{N}}$
        """
    )
    return


@app.cell
def __(pd):
    # Dataframe with AMI data
    ami_test = pd.read_csv('../Loadshape_files/nhec_ami_edited.csv', low_memory=False)
    ami_test['timestamp'] = pd.to_datetime(ami_test['timestamp'])

    # Dataframe with k-mean values
    kmeans_file = pd.read_csv("../Loadshape_files/nhec_2_group_loadshapes.csv")

    # REQUIRES USER INPUT: which graph from k-means has the most buildings in its group? Input the loadshape group #
    res_index = 0

    # Functions to sort the seasons, weekday/ends, and hours of the data's timestamps into different columns
    def season(month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'fall'

    def day_type(day):
        return 'weekday' if day < 5 else 'weekend'

    # For SCE AMI
    # ami_test['season'] = ami_test['timestamp_local'].dt.month.apply(season)
    # ami_test['day_type'] = ami_test['timestamp_local'].dt.dayofweek.apply(day_type)
    # ami_test['hour_test'] = ami_test['timestamp_local'].dt.hour

    ami_test['season'] = ami_test['timestamp'].dt.month.apply(season)
    ami_test['day_type'] = ami_test['timestamp'].dt.dayofweek.apply(day_type)
    ami_test['hour_test'] = ami_test['timestamp'].dt.hour
    return ami_test, day_type, kmeans_file, res_index, season


@app.cell
def __(ami_test):
    ami_test
    return


@app.cell
def __(ami_test):
    # Creating a dictionary for dataframes holding winter power data
    wint_dict = {}
    for _hour_w in range(24):
        for _day_w in ['weekday', 'weekend']:
            wint_combo = (ami_test['hour_test'] == _hour_w) & (ami_test['season'] == 'winter') & (ami_test['day_type'] == _day_w)
            first_letter_w = _day_w[0]
            fifth_letter_w = _day_w[4]
            wint_name = f"win_{first_letter_w}{fifth_letter_w}_{_hour_w}h" # matching name to Loadshape pipeline format
            wint_copy = ami_test[wint_combo].copy()

            # Drop all columns but the power column
            wint_dict[wint_name] = wint_copy.drop(['meter_id', 'timestamp', 'is_dst', \
                                                   'season','day_type','hour_test'], \
                                                  axis=1)

            # For SCE AMI
            # wint_dict[wint_name] = wint_copy.drop(['timestamp_utc', 'timezone_utc', 'is_dst', \
            #                                     'circuit_name','customer_id','timestamp_local','season','day_type','hour_test'], \
            #                                       axis=1)

    # Call the data by using wint_dict['win_wd_23h']
        # All names are formatted as: wint_(we/wd)_(0-23)h
    return (
        fifth_letter_w,
        first_letter_w,
        wint_combo,
        wint_copy,
        wint_dict,
        wint_name,
    )


@app.cell
def __(ami_test):
    # Creating a dictionary for dataframes holding spring power data
    spr_dict = {}
    for _hour_s in range(24):
        for _day_s in ['weekday', 'weekend']:
            spr_combo = (ami_test['hour_test'] == _hour_s) & (ami_test['season'] == 'spring') & (ami_test['day_type'] == _day_s)
            first_letter_sp = _day_s[0]
            fifth_letter_sp = _day_s[4]
            spr_name = f"spr_{first_letter_sp}{fifth_letter_sp}_{_hour_s}h" # matching name to Loadshape pipeline format
            spr_copy = ami_test[spr_combo].copy()

            # Drop all columns but the power column
            spr_dict[spr_name] = spr_copy.drop(['meter_id', 'timestamp', 'is_dst', \
                                                   'season','day_type','hour_test'], \
                                                  axis=1)

            # For SCE AMI
            # spr_dict[spr_name] = spr_copy.drop(['timestamp_utc', 'timezone_utc', 'is_dst', \
            #                                   'circuit_name','customer_id', 'timestamp_local', 'season','day_type','hour_test'], \
            #                                  axis=1)
            

    # Call the data by using spr_dict['spr_wd_23h']
        # All names are formatted as: spr_(we/wd)_(0-23)h
    return (
        fifth_letter_sp,
        first_letter_sp,
        spr_combo,
        spr_copy,
        spr_dict,
        spr_name,
    )


@app.cell
def __(ami_test):
    # Creating a dictionary for dataframes holding summer power data
    sum_dict = {}
    for _hour_su in range(24):
        for _day_su in ['weekday', 'weekend']:
            sum_combo = (ami_test['hour_test'] == _hour_su) & (ami_test['season'] == 'summer') & (ami_test['day_type'] == _day_su)
            first_letter_su = _day_su[0]
            fifth_letter_su = _day_su[4]
            sum_name = f"sum_{first_letter_su}{fifth_letter_su}_{_hour_su}h" # matching name to Loadshape pipeline format
            sum_copy = ami_test[sum_combo].copy()
            
            # Drop all columns but the power column
            sum_dict[sum_name] = sum_copy.drop(['meter_id', 'timestamp', 'is_dst', \
                                                   'season','day_type','hour_test'], \
                                                  axis=1)

            # For SCE AMI
            # sum_dict[sum_name] = sum_copy.drop(['timestamp_utc', 'timezone_utc', 'is_dst', \
            #                                   'circuit_name','customer_id','timestamp_local','season','day_type','hour_test'], \
            #                                  axis=1)

    # Call the data by using sum_dict['sum_we_23h']
        # All names are formatted as: sum_(we/wd)_(0-23)h
    return (
        fifth_letter_su,
        first_letter_su,
        sum_combo,
        sum_copy,
        sum_dict,
        sum_name,
    )


@app.cell
def __(ami_test):
    # Creating a dictionary for dataframes holding fall power data
    fall_dict = {}
    for _hour_f in range(24):
        for _day_f in ['weekday', 'weekend']:
            fall_combo = (ami_test['hour_test'] == _hour_f) & (ami_test['season'] == 'fall') & (ami_test['day_type'] == _day_f)
            first_letter_f = _day_f[0]
            fifth_letter_f = _day_f[4]
            fall_name = f"fal_{first_letter_f}{fifth_letter_f}_{_hour_f}h" # matching name to Loadshape pipeline format
            fall_copy = ami_test[fall_combo].copy()

            # Drop all columns but the power column
            fall_dict[fall_name] = fall_copy.drop(['meter_id', 'timestamp', 'is_dst', \
                                                   'season','day_type','hour_test'], \
                                                  axis=1)

            # For SCE AMI
            # fall_dict[fall_name] = fall_copy.drop(['timestamp_utc', 'timezone_utc', 'is_dst', \
            #                                     'circuit_name','customer_id','timestamp_local','season','day_type','hour_test'], \
            #                                     axis=1)

    #for fall_name, fall_copy in fall_dict.items():
        #print(f"DataFrame {fall_name} created.")

    # Call the data by using fall_dict['fal_wd_23h']
        # All names are formatted as: fal_(we/wd)_(0-23)h
    return (
        fall_combo,
        fall_copy,
        fall_dict,
        fall_name,
        fifth_letter_f,
        first_letter_f,
    )


@app.cell
def __(key, kmeans_file, np, pd, res_index, wint_dict):
    # Winter Standard Deviations

    # K-means has columns with the naming system (win/spr/sum/fal)_(we/wd)_(0-23)h
        # These columns have 1 mean value for each cluster (look at output.png, each cluster has similar loadshapes)
            # Ex: if k-means created 8 groups/clusters, it will have 8 means that we need to find the standard deviation for
    # Each dictionary has dataframes with the naming system (win/spr/sum/fal)_(we/wd)_(0-23)h, which you can call using keys
        # For example, wint_dict has all of the ami data for Dec, Jan, and Feb
        # Each key has the data for a specific hour for weekdays and weekends, so there are 48 keys (2 day types, and 24 hours for each)
        # This loop matches the key with ami data to the column in k-means, so it finds the standard deviation of the ami data for each cluster k-means created to make a new series with (win/spr/sum/fal)_row_means
            # (win/spr/sum/fal)_row_means gives the standard deviation for each cluster k-means created
        # Then, it finds the row the user designated as residential data in res_index as the res_sd_mean for each season
        # The overall seasonal mean of the standard deviations is found in (win/spr/sum/fal)_sd_mean


    # Matching the standard dev dataframe length to the length of the Loadshape dataframe
    dev_winter_data = pd.DataFrame(index=kmeans_file.index) 

    # Loop that matches the k-means value with the matching column in the ami winter dictionary, then stores it into a dataframe
    for _key_w, _ami_winter in wint_dict.items():
        if _key_w in kmeans_file.columns: # if the name of the dataframe is present in the Loadshape file
            # ami_value_w = _ami_winter['energy_value'] # all winter values from ami data that match the Loadshape column name
            ami_value_w = _ami_winter['power'] # all winter values from ami data that match the Loadshape column name
            dev_list_w = []
            for _value_w in kmeans_file[_key_w]: # grabbing column of means from Loadshape
                kmeans_value_w = _value_w  # grabbing column of means from Loadshape
                std_dev_winter = np.sqrt(np.mean((ami_value_w - kmeans_value_w)**2)) # standard deviation formula
                dev_list_w.append(std_dev_winter)
            dev_winter_data[_key_w] = dev_list_w # has all of the stan devs for each Loadshape group for every season, daytype, hour combo
        else:
            print(f"'{key}' not found in k-means DataFrame.")

    # Finding the mean of the standard deviation for the group with residential data (has the most buildings)
    wint_row_means = dev_winter_data.mean(axis=1) # all of the mean standard deviations for each Loadshape group
    wint_res_sd_mean = wint_row_means[res_index] # residential mean standard deviation

    # Finding the mean of the stan dev for all data
    wint_sd_mean = dev_winter_data.mean().mean() # overall mean standard deviation
    return (
        ami_value_w,
        dev_list_w,
        dev_winter_data,
        kmeans_value_w,
        std_dev_winter,
        wint_res_sd_mean,
        wint_row_means,
        wint_sd_mean,
    )


@app.cell
def __(key, kmeans_file, np, pd, res_index, spr_dict):
    # Spring Standard Deviations
    # Matching the standard dev dataframe length to the length of the Loadshape dataframe
    dev_spring_data = pd.DataFrame(index=kmeans_file.index) 

    # Loop that matches the k-means value with the matching column in the ami winter dictionary, then stores it into a dataframe
    for _key_sp, _ami_spr in spr_dict.items():
        if _key_sp in kmeans_file.columns: # if the name of the dataframe is present in the Loadshape file
            # ami_value_sp = _ami_spr['energy_value'] # all spring values from ami data that match the Loadshape column name
            ami_value_sp = _ami_spr['power'] # all spring values from ami data that match the Loadshape column name
            dev_list_sp = []
            for _value_sp in kmeans_file[_key_sp]:  # grabbing column of means from Loadshape
                kmeans_value_sp = _value_sp  # grabbing column of means from Loadshape
                std_dev_spr = np.sqrt(np.mean((ami_value_sp - kmeans_value_sp)**2)) # standard deviation formula
                dev_list_sp.append(std_dev_spr)
            dev_spring_data[_key_sp] = dev_list_sp # has all of the stan devs for each Loadshape group for every season, daytype, hour combo
        else:
            print(f"'{key}' not found in k-means DataFrame.")

    # Finding the mean of the standard deviation for the group with residential data (has the most buildings)
    spr_row_means = dev_spring_data.mean(axis=1) # all of the mean standard deviations for each Loadshape group
    spr_res_sd_mean = spr_row_means[res_index] # residential mean standard deviation

    # Finding the mean of the stan dev for all data
    spr_sd_mean = dev_spring_data.mean().mean() # overall mean standard deviation
    return (
        ami_value_sp,
        dev_list_sp,
        dev_spring_data,
        kmeans_value_sp,
        spr_res_sd_mean,
        spr_row_means,
        spr_sd_mean,
        std_dev_spr,
    )


@app.cell
def __(key, kmeans_file, np, pd, res_index, sum_dict):
    # Summer Standard Deviations
    # Matching the standard dev dataframe length to the length of the Loadshape dataframe
    dev_summer_data = pd.DataFrame(index=kmeans_file.index) 

    # Loop that matches the k-means value with the matching column in the ami winter dictionary, then stores it into a dataframe
    for _key_su, _ami_sum in sum_dict.items():
        if _key_su in kmeans_file.columns: # if the name of the dataframe is present in the Loadshape file
            # ami_value_su = _ami_sum['energy_value'] # all summer values from ami data that match the Loadshape column name
            ami_value_su = _ami_sum['power'] # all summer values from ami data that match the Loadshape column name
            dev_list_su = []
            for _value_su in kmeans_file[_key_su]:  # grabbing column of means from Loadshape
                kmeans_value_su = _value_su  # grabbing column of means from Loadshape
                std_dev_sum = np.sqrt(np.mean((ami_value_su - kmeans_value_su)**2)) # standard deviation formula
                dev_list_su.append(std_dev_sum)
            dev_summer_data[_key_su] = dev_list_su # has all of the stan devs for each Loadshape group for every season, daytype, hour combo
        else:
            print(f"'{key}' not found in k-means DataFrame.")

    # Finding the mean of the standard deviation for the group with residential data (has the most buildings)
    sum_row_means = dev_summer_data.mean(axis=1) # all of the mean standard deviations for each Loadshape group
    sum_res_sd_mean = sum_row_means[res_index] # residential mean standard deviation

    # Finding the mean of the stan dev for all data
    sum_sd_mean = dev_summer_data.mean().mean() # overall mean standard deviation
    return (
        ami_value_su,
        dev_list_su,
        dev_summer_data,
        kmeans_value_su,
        std_dev_sum,
        sum_res_sd_mean,
        sum_row_means,
        sum_sd_mean,
    )


@app.cell
def __(fall_dict, key, kmeans_file, np, pd, res_index):
    # Fall Standard Deviations
    # Matching the standard dev dataframe length to the length of the Loadshape dataframe
    dev_fall_data = pd.DataFrame(index=kmeans_file.index) 

    # Loop that matches the k-means value with the matching column in the ami winter dictionary, then stores it into a dataframe
    for _key_f, _ami_fall in fall_dict.items():
        if _key_f in kmeans_file.columns: # if the name of the dataframe is present in the Loadshape file
            # ami_value_f = _ami_fall['energy_value'] # all fall values from ami data that match the Loadshape column name
            ami_value_f = _ami_fall['power'] # all fall values from ami data that match the Loadshape column name
            dev_list_f = []
            for _value_f in kmeans_file[_key_f]:  # grabbing column of means from Loadshape
                kmeans_value_f = _value_f  # grabbing column of means from Loadshape
                std_dev_fall = np.sqrt(np.mean((ami_value_f - kmeans_value_f)**2)) # standard deviation formula
                dev_list_f.append(std_dev_fall)
            dev_fall_data[_key_f] = dev_list_f # has all of the stan devs for each Loadshape group for every season, daytype, hour combo
        else:
            print(f"'{key}' not found in k-means DataFrame.")

    # Finding the mean of the standard deviation for the group with residential data (has the most buildings)
    fall_row_means = dev_fall_data.mean(axis=1) # all of the mean standard deviations for each Loadshape group
    fall_res_sd_mean = fall_row_means[res_index] # residential mean standard deviation

    # Finding the mean of the stan dev for all data
    fall_sd_mean = dev_fall_data.mean().mean() # overall mean standard deviation
    return (
        ami_value_f,
        dev_fall_data,
        dev_list_f,
        fall_res_sd_mean,
        fall_row_means,
        fall_sd_mean,
        kmeans_value_f,
        std_dev_fall,
    )


@app.cell
def __(
    fall_res_sd_mean,
    fall_sd_mean,
    kmeans_file,
    np,
    res_index,
    spr_res_sd_mean,
    spr_sd_mean,
    sum_res_sd_mean,
    sum_sd_mean,
    wint_res_sd_mean,
    wint_sd_mean,
):
    print(f'This data represents the means of the standard deviations when the Loadshape pipeline created {len(kmeans_file.index)} groups.')
    print(f'The residential data reflects the data from group #{res_index}')
    print ('')

    # Residential means of standard deviations for all seasons
    print(f'Winter Residential SD Mean: {wint_res_sd_mean}')
    print(f'Spring Residential SD Mean: {spr_res_sd_mean}')
    print(f'Summer Residential SD Mean: {sum_res_sd_mean}')
    print(f'Fall Residential SD Mean: {fall_res_sd_mean}')

    # Means of standard deviations for all seasons
    print(f'Winter SD Mean: {wint_sd_mean}')
    print(f'Spring SD Mean: {spr_sd_mean}')
    print(f'Summer SD Mean: {sum_sd_mean}')
    print(f'Fall SD Mean: {fall_sd_mean}')

    print ('')

    # Residential mean of all SD
    residential_sd_mean = np.mean([wint_res_sd_mean, spr_res_sd_mean, sum_res_sd_mean, fall_res_sd_mean])
    print(f'Residential SD Mean for {len(kmeans_file.index)} groups: {residential_sd_mean}, COPY THIS')

    # Mean of all SD
    sd_mean = np.mean([wint_sd_mean, spr_sd_mean, sum_sd_mean, fall_sd_mean])
    print(f'Overall SD Mean for {len(kmeans_file.index)} groups: {sd_mean}, COPY THIS')
    return residential_sd_mean, sd_mean


@app.cell
def __():
    # Numbers for SCE AMI Data
    group_num_sce = list(range(2, 41, 2))
    res_std_sce = [77.45636168807386, 78.20920935652418, 78.37951422305528, 78.4317060966416, 78.44732158031563, \
               78.5100262706137, 79.00512565841025, 79.00901393003964, 79.02006604798258, 79.10363424937404, \
               78.31917924949451, 77.92872779684275, 77.94440954901535, 78.71404695361632, 77.9141939190178, \
               78.32912042314184, 78.31883694618378, 78.7129747592283, 78.85050820549716, 78.74453476913231]
    data_std_sce = [226.87962545172394, 297.2280825103721, 320.48016541965245, 366.75076000416476, 318.931551548227, \
                351.84192685858176, 298.4413038242042, 294.29457305102454, 298.34041770149895, 273.39211008863015, \
                255.28224423911004, 252.76459282984752, 248.357133227526, 239.6570981634439, 239.682075770267, \
                231.83263397541486, 231.01050584164645, 219.4379313787641, 220.59047469354974, 213.97989620875032]
    return data_std_sce, group_num_sce, res_std_sce


@app.cell
def __():
    # Numbers for NHEC AMI Data
    group_num_nhec = list(range(2, 21, 2))
    res_std_nhec = [18.548194190212875, 19.05028697037593, 19.10591284406399, 19.16981681712639, 19.253685501274717, \
                   19.09164097926809, 19.10486311283183, 19.264825181693748, 19.107306220338256, 19.291648956883797]
    data_std_nhec = [50.24774185230634, 46.226255201037766, 38.425821680942136, 34.12801192125563, 31.799946139492825, \
                    29.65106262233722, 28.17356431590144, 27.0144923334527, 26.141283832779955, 25.42678527356563]
    return data_std_nhec, group_num_nhec, res_std_nhec


@app.cell
def __(group_num_sce, pd, plt, res_std_sce):
    # Plotting the means of the standard deviations from the SCE AMI residential data
    stdev_sce_res = pd.Series(res_std_sce,index=group_num_sce)

    plt.figure(figsize=(8, 5))
    stdev_sce_res.plot(marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xticks(range(min(group_num_sce), max(group_num_sce)+1, 2))
    plt.title('Means of Standard Deviations, SCE AMI - Residential', fontsize=17)
    plt.xlabel('Number of groups (k)', fontsize=17)
    plt.ylabel('Mean (kW)', fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0,100])
    plt.gca()
    return stdev_sce_res,


@app.cell
def __(data_std_sce, group_num_sce, pd, plt):
    # Plotting the means of the standard deviations from all of the SCE AMI data
    stdev_sce_data = pd.Series(data_std_sce,index=group_num_sce)

    plt.figure(figsize=(8, 5))
    stdev_sce_data.plot(marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
    plt.xticks(range(min(group_num_sce), max(group_num_sce)+1, 2))
    plt.title('Means of Standard Deviations, SCE AMI', fontsize=17)
    plt.xlabel('Number of groups (k)', fontsize=17)
    plt.ylabel('Mean (kW)', fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.gca()
    return stdev_sce_data,


@app.cell
def __(group_num_nhec, pd, plt, res_std_nhec):
    # Plotting the means of the standard deviations from the NHEC AMI residential data
    stdev_nhec_res = pd.Series(res_std_nhec,index=group_num_nhec)

    plt.figure(figsize=(8, 5))
    stdev_nhec_res.plot(marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xticks(range(min(group_num_nhec), max(group_num_nhec)+1, 2))
    plt.title('Means of Standard Deviations, NHEC AMI - Residential', fontsize=17)
    plt.xlabel('Number of groups (k)', fontsize=17)
    plt.ylabel('Mean (kW)', fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0,50])
    plt.gca()
    return stdev_nhec_res,


@app.cell
def __(data_std_nhec, group_num_nhec, pd, plt):
    # Plotting the means of the standard deviations from all of the NHEC AMI data
    stdev_nhec_data = pd.Series(data_std_nhec,index=group_num_nhec)

    plt.figure(figsize=(8, 5))
    stdev_nhec_data.plot(marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
    plt.xticks(range(min(group_num_nhec), max(group_num_nhec)+1, 2))
    plt.title('Means of Standard Deviations, NHEC AMI', fontsize=17)
    plt.xlabel('Number of groups (k)', fontsize=17)
    plt.ylabel('Mean (kW)', fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.ylim([0,60])
    plt.gca()
    return stdev_nhec_data,


@app.cell
def __(mo):
    mo.md(
        r"""
        # GISMo Forecast Data
        - The metadata files were used to find the building count for each house type. The data shown is power per square feet for an average household over an average day.
        """
    )
    return


@app.cell
def __(pd):
    resstock_bldgs = pd.read_csv("../metadata/resstock_metadata.csv")
    orangecty_resstock = resstock_bldgs.loc[resstock_bldgs['county'] == 'G0600590'] # Code for Orange County
    orange_mobile = orangecty_resstock.loc[orangecty_resstock['building_type'] == 'Mobile Home']
    orange_multi24 = orangecty_resstock.loc[orangecty_resstock['building_type'] == 'Multi-Family with 2 - 4 Units']
    orange_multi5 = orangecty_resstock.loc[orangecty_resstock['building_type'] == 'Multi-Family with 5+ Units']
    orange_sin_att = orangecty_resstock.loc[orangecty_resstock['building_type'] == 'Single-Family Attached']
    orange_sin_det = orangecty_resstock.loc[orangecty_resstock['building_type'] == 'Single-Family Detached']

    print(f'There are *{len(orange_mobile["bldg_id"].unique())}* mobile homes in the Resstock Data.')
    print(f'There are *{len(orange_multi24["bldg_id"].unique())}* multi-family homes with 2-4 units in the Resstock Data.')
    print(f'There are *{len(orange_multi5["bldg_id"].unique())}* multi-family homes with 5+ units in the Resstock Data.')
    print(f'There are *{len(orange_sin_att["bldg_id"].unique())}* single-family attached homes in the Resstock Data.')
    print(f'There are *{len(orange_sin_det["bldg_id"].unique())}* single-family detached homes in the Resstock Data.')
    return (
        orange_mobile,
        orange_multi24,
        orange_multi5,
        orange_sin_att,
        orange_sin_det,
        orangecty_resstock,
        resstock_bldgs,
    )


@app.cell
def __(mo):
    # Plotting Resstock GISMo Forecast Data w/ Marimo features
    plots = ['Average Day - Mobile Homes', 'Average Day - Multi-Family Homes w/ 2-4 Units', 'Average Day - Multi-Family Homes w/ 5+ Units', 'Average Day - Single-Family Attached Homes', 'Average Day - Single-Family Detached Homes', 'Sum of All Home Types', 'Avg of All Home Types']
    selected_plot = mo.ui.dropdown(plots,plots[0])
    selected_plot
    return plots, selected_plot


@app.cell
def __(np, pd):
    # Normalized GISMo Forecast Data (kW/sf)
    gismo_mobile = pd.read_csv('../resstock_normalized_power/orange_county_mobile_home_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_multi_24 = pd.read_csv('../resstock_normalized_power/orange_county_multi-family_with_2_-_4_units_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_multi_5 = pd.read_csv('../resstock_normalized_power/orange_county_multi-family_with_5plus_units_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_single_att = pd.read_csv('../resstock_normalized_power/orange_county_single-family_attached_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_single_det = pd.read_csv('../resstock_normalized_power/orange_county_single-family_detached_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])

    # Finding average by dividing data by number of houses used in the simulations, found from metadata
    mobile_data = (gismo_mobile['Total Electricity [kW/sf]']) / 120
    multi24_data = (gismo_multi_24['Total Electricity [kW/sf]']) / 387
    multi5_data = (gismo_multi_5['Total Electricity [kW/sf]']) / 1105
    sing_att_data = (gismo_single_att['Total Electricity [kW/sf]']) / 540
    sing_det_data = (gismo_single_det['Total Electricity [kW/sf]']) / 2275

    # Finding the sum of all house types
    gismo_normalized_sum = mobile_data + multi24_data + multi5_data + sing_att_data + sing_det_data
    hours_gismo_norm = list(range(0, 24))

    # Finding the overall average of all house types
    avg_gismo_stacked = np.stack([mobile_data, multi24_data, multi5_data, sing_att_data, sing_det_data])
    avg_gismo_norm = np.mean(avg_gismo_stacked, axis=0)
    return (
        avg_gismo_norm,
        avg_gismo_stacked,
        gismo_mobile,
        gismo_multi_24,
        gismo_multi_5,
        gismo_normalized_sum,
        gismo_single_att,
        gismo_single_det,
        hours_gismo_norm,
        mobile_data,
        multi24_data,
        multi5_data,
        sing_att_data,
        sing_det_data,
    )


@app.cell
def __(
    avg_gismo_norm,
    gismo_mobile,
    gismo_multi_24,
    gismo_multi_5,
    gismo_normalized_sum,
    gismo_single_att,
    gismo_single_det,
    hours_gismo_norm,
    mobile_data,
    multi24_data,
    multi5_data,
    pd,
    plots,
    plt,
    selected_plot,
    sing_att_data,
    sing_det_data,
):
    # Loop that prints the plots for each ResStock plot when the user uses the Marimo button feature above
    if selected_plot.value == plots[0]:
        res_plot = pd.Series(mobile_data,index=gismo_mobile['hour'])
        title = 'Average Day from the GISMo Forecast for a Mobile Home'
    elif selected_plot.value == plots[1]:
        res_plot = pd.Series(multi24_data,index=gismo_multi_24['hour'])
        title = 'Average Day from the GISMo Forecast for a Multi-Family Home w/ 2-4 Units'
    elif selected_plot.value == plots[2]:
        res_plot = pd.Series(multi5_data,index=gismo_multi_5['hour'])
        title = 'Average Day from the GISMo Forecast for a Multi-Family Home w/ 5+ Units'
    elif selected_plot.value == plots[3]:
        res_plot = pd.Series(sing_att_data,index=gismo_single_att['hour'])
        title = 'Average Day from the GISMo Forecast for a Single-Family Attached Home'
    elif selected_plot.value == plots[4]:
        res_plot = pd.Series(sing_det_data,index=gismo_single_det['hour'])
        title = 'Average Day from the GISMo Forecast for a Single-Family Detached Home'
    elif selected_plot.value == plots[5]:
        res_plot = pd.Series(gismo_normalized_sum,index=hours_gismo_norm)
        title = 'Average Day from the GISMo Forecast, Sum of All Home Types'
    elif selected_plot.value == plots[6]:
        res_plot = pd.Series(avg_gismo_norm,index=hours_gismo_norm)
        title = 'Average Day from the GISMo Forecast, Avg of All Home Types'
    else:
        print('Option is not available')

    plt.figure(figsize=(20, 10))
    res_plot.plot(color='blue', linestyle='-', linewidth=2, markersize=8)
    plt.legend([])
    plt.grid(True)
    plt.xticks(range(min(res_plot.index), max(res_plot.index)+1, 2), fontsize=16)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=30)
    plt.xlabel('Hours', fontsize=30)
    plt.ylabel('Total Power [kW/sf]', fontsize=30)
    return res_plot, title


@app.cell
def __(pd):
    comstock_bldgs = pd.read_csv("../metadata/comstock_metadata.csv")
    orangecty_comstock = comstock_bldgs.loc[comstock_bldgs['county'] == 'G0600590']
    orange_warehouse = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'Warehouse']
    orange_soffice = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'SmallOffice']
    orange_retmall = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'RetailStripmall']
    orange_retalone = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'RetailStandalone']
    orange_fullrest = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'FullServiceRestaurant']
    orange_quickrest = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'QuickServiceRestaurant']
    orange_moffice = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'MediumOffice']
    orange_primschool = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'PrimarySchool']
    orange_seconschool = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'SecondarySchool']
    orange_outpat = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'Outpatient']
    orange_shotel = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'SmallHotel']
    orange_lhotel = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'LargeHotel']
    orange_loffice = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'LargeOffice']
    orange_hospital = orangecty_comstock.loc[orangecty_comstock['building_type'] == 'Hospital']

    print(f'There are *{len(orange_shotel["bldg_id"].unique())}* small hotels in the Comstock Data.')
    print(f'There are *{len(orange_lhotel["bldg_id"].unique())}* large hotels in the Comstock Data.')
    print(f'There are *{len(orange_soffice["bldg_id"].unique())}* small offices in the Comstock Data.')
    print(f'There are *{len(orange_moffice["bldg_id"].unique())}* medium offices in the Comstock Data.')
    print(f'There are *{len(orange_loffice["bldg_id"].unique())}* large offices in the Comstock Data.')
    print(f'There are *{len(orange_primschool["bldg_id"].unique())}* primary schools in the Comstock Data.')
    print(f'There are *{len(orange_seconschool["bldg_id"].unique())}* secondary schools in the Comstock Data.')
    print(f'There are *{len(orange_retalone["bldg_id"].unique())}* retail stand alones in the Comstock Data.')
    print(f'There are *{len(orange_retmall["bldg_id"].unique())}* retail strip malls in the Comstock Data.')
    print(f'There are *{len(orange_warehouse["bldg_id"].unique())}* warehouses in the Comstock Data.')
    print(f'There are *{len(orange_fullrest["bldg_id"].unique())}* full service restaurants in the Comstock Data.')
    print(f'There are *{len(orange_quickrest["bldg_id"].unique())}* quick service restaurants in the Comstock Data.')
    print(f'There are *{len(orange_hospital["bldg_id"].unique())}* hospitals in the Comstock Data.')
    print(f'There are *{len(orange_outpat["bldg_id"].unique())}* outpatient buildings in the Comstock Data.')
    return (
        comstock_bldgs,
        orange_fullrest,
        orange_hospital,
        orange_lhotel,
        orange_loffice,
        orange_moffice,
        orange_outpat,
        orange_primschool,
        orange_quickrest,
        orange_retalone,
        orange_retmall,
        orange_seconschool,
        orange_shotel,
        orange_soffice,
        orange_warehouse,
        orangecty_comstock,
    )


@app.cell
def __(np, pd):
    # Normalized GISMo Comstock Data
    gismo_small_hotel = pd.read_csv('../comstock_normalized_power/com_orange_county_smallhotel_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_large_hotel = pd.read_csv('../comstock_normalized_power/com_orange_county_largehotel_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_small_office = pd.read_csv('../comstock_normalized_power/com_orange_county_smalloffice_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_med_office = pd.read_csv('../comstock_normalized_power/com_orange_county_mediumoffice_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_large_office = pd.read_csv('../comstock_normalized_power/com_orange_county_largeoffice_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_prim_school = pd.read_csv('../comstock_normalized_power/com_orange_county_primaryschool_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_second_school = pd.read_csv('../comstock_normalized_power/com_orange_county_secondaryschool_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_retail_alone = pd.read_csv('../comstock_normalized_power/com_orange_county_retailstandalone_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_retail_mall = pd.read_csv('../comstock_normalized_power/com_orange_county_retailstripmall_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_warehouse = pd.read_csv('../comstock_normalized_power/com_orange_county_warehouse_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_full_rest = pd.read_csv('../comstock_normalized_power/com_orange_county_fullservicerestaurant_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_quick_rest = pd.read_csv('../comstock_normalized_power/com_orange_county_quickservicerestaurant_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_hospital = pd.read_csv('../comstock_normalized_power/com_orange_county_hospital_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])
    gismo_outpatient = pd.read_csv('../comstock_normalized_power/com_orange_county_outpatient_total_energy_normalized.csv', usecols=['hour', 'Total Electricity [kW/sf]'])


    # Finding average by dividing data by number of houses used in data
    small_hotel = (gismo_small_hotel['Total Electricity [kW/sf]']) / 41
    large_hotel = (gismo_large_hotel['Total Electricity [kW/sf]']) / 16
    small_office = (gismo_small_office['Total Electricity [kW/sf]']) / 563
    med_office = (gismo_med_office['Total Electricity [kW/sf]']) / 99
    large_office = (gismo_large_office['Total Electricity [kW/sf]']) / 11
    prim_school = (gismo_prim_school['Total Electricity [kW/sf]']) / 47
    second_school = (gismo_second_school['Total Electricity [kW/sf]']) / 23
    retail_alone = (gismo_retail_alone['Total Electricity [kW/sf]']) / 366
    retail_mall = (gismo_retail_mall['Total Electricity [kW/sf]']) / 484
    warehouses = (gismo_warehouse['Total Electricity [kW/sf]']) / 1242
    full_rest = (gismo_full_rest['Total Electricity [kW/sf]']) / 131
    quick_rest = (gismo_quick_rest['Total Electricity [kW/sf]']) / 96
    hospital = (gismo_hospital['Total Electricity [kW/sf]']) / 4
    outpatient = (gismo_outpatient['Total Electricity [kW/sf]']) / 69

    # Finding the overall average of all house types
    avg_comstock_stacked = np.stack([small_hotel, large_hotel, small_office, med_office, large_office, prim_school, second_school, \
                                     retail_alone, retail_mall, warehouses, full_rest, quick_rest, hospital, outpatient])
    avg_comstock_norm = np.mean(avg_comstock_stacked, axis=0)
    return (
        avg_comstock_norm,
        avg_comstock_stacked,
        full_rest,
        gismo_full_rest,
        gismo_hospital,
        gismo_large_hotel,
        gismo_large_office,
        gismo_med_office,
        gismo_outpatient,
        gismo_prim_school,
        gismo_quick_rest,
        gismo_retail_alone,
        gismo_retail_mall,
        gismo_second_school,
        gismo_small_hotel,
        gismo_small_office,
        gismo_warehouse,
        hospital,
        large_hotel,
        large_office,
        med_office,
        outpatient,
        prim_school,
        quick_rest,
        retail_alone,
        retail_mall,
        second_school,
        small_hotel,
        small_office,
        warehouses,
    )


@app.cell
def __(mo):
    # Marimo button feature for ComStock data
    plots_comstock = ['Average Day - Small Hotel', 'Average Day - Large Hotel', 'Average Day - Small Office', 'Average Day - Medium Office', 'Average Day - Large Office', 'Average Day - Primary School', 'Average Day - Secondary School', 'Average Day - Retail Stand Alone', 'Average Day - Retail Strip Mall', 'Average Day - Warehouse', 'Average Day - Full Service Restaurant', 'Average Day - Quick Service Restaurant', 'Average Day - Hospital', 'Average Day - Outpatient', 'Avg of All Home Types']
    selected_plot_comstock = mo.ui.dropdown(plots_comstock,plots_comstock[0])
    selected_plot_comstock
    return plots_comstock, selected_plot_comstock


@app.cell
def __(
    avg_comstock_norm,
    full_rest,
    gismo_full_rest,
    gismo_hospital,
    gismo_large_hotel,
    gismo_large_office,
    gismo_med_office,
    gismo_outpatient,
    gismo_prim_school,
    gismo_quick_rest,
    gismo_retail_alone,
    gismo_retail_mall,
    gismo_second_school,
    gismo_small_hotel,
    gismo_small_office,
    gismo_warehouse,
    hospital,
    hours_gismo_norm,
    large_hotel,
    large_office,
    med_office,
    outpatient,
    pd,
    plots_comstock,
    plt,
    prim_school,
    quick_rest,
    retail_alone,
    retail_mall,
    second_school,
    selected_plot_comstock,
    small_hotel,
    small_office,
    warehouses,
):
    # Loop for button functionality
    if selected_plot_comstock.value == plots_comstock[0]:
        res_plot_comstock = pd.Series(small_hotel,index=gismo_small_hotel['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Small Hotel'
    elif selected_plot_comstock.value == plots_comstock[1]:
        res_plot_comstock = pd.Series(large_hotel,index=gismo_large_hotel['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Large Hotel'
    elif selected_plot_comstock.value == plots_comstock[2]:
        res_plot_comstock = pd.Series(small_office,index=gismo_small_office['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Small Office'
    elif selected_plot_comstock.value == plots_comstock[3]:
        res_plot_comstock = pd.Series(med_office,index=gismo_med_office['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Medium Office'
    elif selected_plot_comstock.value == plots_comstock[4]:
        res_plot_comstock = pd.Series(large_office,index=gismo_large_office['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Large Office'
    elif selected_plot_comstock.value == plots_comstock[5]:
        res_plot_comstock = pd.Series(prim_school,index=gismo_prim_school['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Primary School'
    elif selected_plot_comstock.value == plots_comstock[6]:
        res_plot_comstock = pd.Series(second_school,index=gismo_second_school['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Secondary School'
    elif selected_plot_comstock.value == plots_comstock[7]:
        res_plot_comstock = pd.Series(retail_alone,index=gismo_retail_alone['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Retail Stand Alone'
    elif selected_plot_comstock.value == plots_comstock[8]:
        res_plot_comstock = pd.Series(retail_mall,index=gismo_retail_mall['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Retail Strip Mall'
    elif selected_plot_comstock.value == plots_comstock[9]:
        res_plot_comstock = pd.Series(warehouses,index=gismo_warehouse['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Warehouse'
    elif selected_plot_comstock.value == plots_comstock[10]:
        res_plot_comstock = pd.Series(full_rest,index=gismo_full_rest['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Full Service Restaurant'
    elif selected_plot_comstock.value == plots_comstock[11]:
        res_plot_comstock = pd.Series(quick_rest,index=gismo_quick_rest['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Quick Service Restaurant'
    elif selected_plot_comstock.value == plots_comstock[12]:
        res_plot_comstock = pd.Series(hospital,index=gismo_hospital['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for a Hospital'
    elif selected_plot_comstock.value == plots_comstock[13]:
        res_plot_comstock = pd.Series(outpatient,index=gismo_outpatient['hour'])
        title_comstock = 'Average Day from the GISMo Forecast for an Outpatient Building'
    elif selected_plot_comstock.value == plots_comstock[14]:
        res_plot_comstock = pd.Series(avg_comstock_norm,index=hours_gismo_norm)
        title_comstock = 'Average Day from the GISMo Forecast, Avg of All Home Types'
    else:
        print('Option is not available')

    plt.figure(figsize=(20, 10))
    res_plot_comstock.plot(color='blue', linestyle='-', linewidth=2, markersize=8)
    plt.legend([])
    plt.grid(True)
    plt.xticks(range(min(res_plot_comstock.index), max(res_plot_comstock.index)+1, 2), fontsize=16)
    plt.yticks(fontsize=20)
    plt.title(title_comstock, fontsize=30)
    plt.xlabel('Hours', fontsize=30)
    plt.ylabel('Total Power [kW/sf]', fontsize=30)
    return res_plot_comstock, title_comstock


@app.cell
def __(mo):
    mo.md(
        r"""
        # House Data for Orange County (Normalizing AMI Data by SF)
        - This data was used to find the average square footage for a residential house in Orange County. The most common house type was "IECC", which has a mean square footage of 3449.54 sq ft.
        - Average house data for New Hampshire was acquired from
            - https://www.eia.gov/consumption/residential/data/2020/index.php?view=characteristics
            - https://www.eia.gov/consumption/residential/data/2020/state/pdf/State%20Square%20Footage.pdf
            - The avg sq ft per housing unit, total was used: 2,102 sq ft
        """
    )
    return


@app.cell
def __(pd):
    # House Data for Orange County
    orange_cty = pd.read_csv("../metadata/CA-Orange.csv.zip")
    orange_cty
    return orange_cty,


@app.cell
def __(np, orange_cty):
    iecc = orange_cty.loc[orange_cty['BuildingType'] == 'IECC'] # most common house type
    print(f'The mean value for IECC buildings in Orange County is {np.mean(iecc["FloorArea"])}.')
    print(f'The mean value for all buildings in Orange County is {np.mean(orange_cty["FloorArea"])}.')
    return iecc,


@app.cell
def __(mo):
    mo.md(
        r"""
        # AMI Data: Average Days w/ Best Fit Groups and RMSE
        - The root mean squared error (RMSE) was found using the formula: $\sqrt{\displaystyle\sum_{i=1}^n \frac{(y_{pred} - y_{actual})^{2}}{n}}$
        - The RMSE was then divided by the standard deviation of the data and the Loadshape averages to find the normalized RMSE (NRMSE). The NRMSE can also be found by dividing by the mean of the data, and that number was also calculated.
        """
    )
    return


@app.cell
def __(np):
    # RMSE Function
    def find_rmse(actual, predicted):
        return np.sqrt(np.mean((actual - predicted) ** 2))
    return find_rmse,


@app.cell
def __(gismo_mobile, mobile_data, pd, plt):
    # 2 GROUPS, SCE AMI
    # Plotting SCE AMI data after it has been sorted by the Loadshape pipeline
    sce_ami_kmeans_2s = pd.read_csv('../Loadshape_files/sce_2_group_loadshapes.csv')

    # What group is the residential data in?
    res_group_2s = 0

    # Isolating residential data and removing the group row
    group_2s = sce_ami_kmeans_2s.iloc[res_group_2s] # acquire residential kmeans data
    ami_format_2s = group_2s.drop(group_2s.index[0]) # remove row designating group #

    # Finding the mean of every hour to create an average day
    hour_str_2s = ami_format_2s.index.str.split('_').str[-1].str[:-1] # acquiring hour number from column names
    hours_2s = ami_format_2s.groupby(hour_str_2s).mean() / 3449.5423072719786 # finding hourly means and normalizing by sqft w/ IECC mean
    hours_2s.index = hours_2s.index.astype(int)
    ami_avg_day_2s = hours_2s.sort_index() # sorting hours to be ascending

    #Plotting AMI Data over the Resstock graph
    fig, ami = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)

    # SCE AMI Graph
    ami.plot(ami_avg_day_2s.index, ami_avg_day_2s.values, linestyle='-.', color='r', label='SCE AMI')
    ami.set_title('Average Day from SCE AMI [2 groups] and Resstock Data', fontsize=30)
    ami.set_xlabel('Hours', fontsize=25)
    ami.set_ylabel('Mean Power for an Avg House, Orange County (kW/sf)', fontsize=20)
    ami.tick_params(axis='y', labelsize=17)
    ami.set_xticks(range(min(ami_avg_day_2s.index), max(ami_avg_day_2s.index)+1, 2))
    ami.set_xticklabels(range(min(ami_avg_day_2s.index), max(ami_avg_day_2s.index)+1, 2), fontsize=17)
    ami.grid(True)

    # Overlaying Resstock graph
    resstock_graph = ami.twinx()
    res_plot_ex = pd.Series((mobile_data),index=gismo_mobile['hour'])
    resstock_graph.plot(res_plot_ex.index, res_plot_ex.values, color='blue', linestyle='-', linewidth=2, markersize=8, label='Resstock, Mobile Home')
    resstock_graph.set_ylabel('Mean Power for an Avg Resstock Mobile Home (kW/sf)', fontsize=20)
    resstock_graph.tick_params(axis='y', labelsize=17)

    # Legend information and showing graph for Marimo app
    ami_data, ami_labels = ami.get_legend_handles_labels()
    resstockgraph, resstock_label = resstock_graph.get_legend_handles_labels()
    ami.legend(ami_data + resstockgraph, ami_labels + resstock_label, fontsize=25)
    plt.gca()
    return (
        ami,
        ami_avg_day_2s,
        ami_data,
        ami_format_2s,
        ami_labels,
        fig,
        group_2s,
        hour_str_2s,
        hours_2s,
        res_group_2s,
        res_plot_ex,
        resstock_graph,
        resstock_label,
        resstockgraph,
        sce_ami_kmeans_2s,
    )


@app.cell
def __(ami_avg_day_2s, gismo_med_office, med_office, pd, plt):
    # Plotting AMI Data over the ComStock graph
    fig3, ami3 = plt.subplots()
    fig3.set_figheight(10)
    fig3.set_figwidth(20)


    # SCE AMI Graph
    ami3.plot(ami_avg_day_2s.index, ami_avg_day_2s.values, linestyle='-.', color='r', label='SCE AMI')
    ami3.set_title('Average Day from SCE AMI [2 groups] and Comstock Data', fontsize=30)
    ami3.set_xlabel('Hours', fontsize=25)
    ami3.set_ylabel('Mean Power for an Avg House, Orange County (kW/sf)', fontsize=20)
    ami3.tick_params(axis='y', labelsize=17)
    ami3.set_xticks(range(min(ami_avg_day_2s.index), max(ami_avg_day_2s.index)+1, 2))
    ami3.set_xticklabels(range(min(ami_avg_day_2s.index), max(ami_avg_day_2s.index)+1, 2), fontsize=17)
    ami3.grid(True)

    # Overlaying ComStock graph
    comstock_graph2 = ami3.twinx()
    comstock_data = pd.Series(med_office,index=gismo_med_office['hour'])
    comstock_graph2.plot(comstock_data.index, comstock_data.values, color='blue', linestyle='-', linewidth=2, markersize=8, label='Comstock, Medium Office')
    comstock_graph2.set_ylabel('Mean Power for an Avg Comstock Medium Office (kW/sf)', fontsize=20)
    comstock_graph2.tick_params(axis='y', labelsize=17)

    # Legend information and showing graph for Marimo app
    ami_data3, ami_labels3 = ami3.get_legend_handles_labels()
    comstockgraph2, comstock_label2 = comstock_graph2.get_legend_handles_labels()
    ami3.legend(ami_data3 + comstockgraph2, ami_labels3 + comstock_label2, fontsize=25)
    plt.gca()
    return (
        ami3,
        ami_data3,
        ami_labels3,
        comstock_data,
        comstock_graph2,
        comstock_label2,
        comstockgraph2,
        fig3,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        - When comparing SCE AMI data to the **Resstock** GISMo averages, the statistics are:
            - RMSE for the GISMo forecast compared to the k-means 2 group data is 0.00422
            - NRMSE divided by the standard deviation for 2 k-mean groups is 4.96
            - NRMSE divided by the mean for 2 k-mean groups is 0.922
        - When comparing SCE AMI data to the **Comstock** GISMo averages, the statistics are:
            - RMSE for the GISMo forecast compared to the k-means 2 group data is 0.00388
            - NRMSE divided by standard deviation for 2 k-mean groups is 4.56
            - NRMSE divided by the mean for 2 k-mean groups is 0.849
        """
    )
    return


@app.cell
def __(ami_avg_day_2s, avg_gismo_norm, find_rmse, np):
    # Comparing SCE and ResStock
    rmse_g24 = find_rmse(ami_avg_day_2s.values, avg_gismo_norm)

    rsme_std_g24_ami = rmse_g24 / (np.std(ami_avg_day_2s.values))
    rsme_std_g24_gismo = rmse_g24 / (np.std(avg_gismo_norm))

    print(f'The RMSE for the GISMo forecast compared to the k-means 2 group data is {rmse_g24}.')
    print("")
    print(f'The NRMSE divided by standard deviation for 2 k-mean groups is {rsme_std_g24_ami}.')
    print(f'The GISMo RMSE divided by standard deviation for 2 k-mean groups is {rsme_std_g24_gismo}.')
    print ("")
    print(f'The NRMSE divided by the mean for Resstock is {rmse_g24 / (np.mean(ami_avg_day_2s.values))}.')
    return rmse_g24, rsme_std_g24_ami, rsme_std_g24_gismo


@app.cell
def __(ami_avg_day_2s, avg_comstock_norm, find_rmse, np):
    # Comparing SCE and ComStock
    rmse_g24c = find_rmse(ami_avg_day_2s.values, avg_comstock_norm)

    rsme_std_g24_amic = rmse_g24c / (np.std(ami_avg_day_2s.values))
    rsme_std_g24_gismoc = rmse_g24c / (np.std(avg_comstock_norm))

    print(f'The RMSE for the GISMo forecast compared to the k-means 2 group data is {rmse_g24c}.')
    print("")
    print(f'The NRMSE divided by standard deviation for 2 k-mean groups is {rsme_std_g24_amic}.')
    print(f'The GISMo RMSE divided by standard deviation for 2 k-mean groups is {rsme_std_g24_gismoc}.')
    print("")
    print(f'The NRMSE divided by the mean for Comstock is {rmse_g24c / (np.mean(ami_avg_day_2s.values))}')
    return rmse_g24c, rsme_std_g24_amic, rsme_std_g24_gismoc


@app.cell
def __(gismo_med_office, med_office, pd, plt):
    # 2 GROUPS
    # Plotting NHEC AMI data after it has been sorted by the Loadshape pipeline
    nhec_ami_kmeans_2 = pd.read_csv('../Loadshape_files/nhec_2_group_loadshapes.csv')

    # What group is the residential data in?
    res_group_2 = 0

    # Isolating residential data and removing the group row
    group_2 = nhec_ami_kmeans_2.iloc[res_group_2]
    ami_format_2 = group_2.drop(group_2.index[0])

    # Finding the mean of every hour to create an average day
    hour_str_2 = ami_format_2.index.str.split('_').str[-1].str[:-1]
    hours_2 = ami_format_2.groupby(hour_str_2).mean() / 2102 # normalizing by sq ft with state mean
    hours_2.index = hours_2.index.astype(int)
    ami_avg_day_2 = hours_2.sort_index()

    #Plotting AMI Data over the Comstock graph
    fig2, ami2 = plt.subplots()
    fig2.set_figheight(10)
    fig2.set_figwidth(20)

    # NHEC AMI Graph
    ami2.plot(ami_avg_day_2.index, ami_avg_day_2.values, linestyle='-.', color='r', label='NHEC AMI')
    ami2.set_title('Average Day from NHEC AMI [2 groups] and Comstock Data', fontsize=30)
    ami2.set_xlabel('Hours', fontsize=25)
    ami2.set_ylabel('Mean Power for an Avg House, New Hampshire (kW/sf)', fontsize=20)
    ami2.tick_params(axis='y', labelsize=17)
    ami2.set_xticks(range(min(ami_avg_day_2.index), max(ami_avg_day_2.index)+1, 2))
    ami2.set_xticklabels(range(min(ami_avg_day_2.index), max(ami_avg_day_2.index)+1, 2), fontsize=17)
    ami2.grid(True)

    # Overlaying Comstock graph
    comstock_graph = ami2.twinx()
    res_plot_comstock_ex = pd.Series(med_office,index=gismo_med_office['hour'])
    comstock_graph.plot(res_plot_comstock_ex.index, res_plot_comstock_ex.values, color='blue', linestyle='-', linewidth=2, markersize=8, label='Comstock, Medium Office')
    comstock_graph.set_ylabel('Mean Power for an Avg Comstock Medium Office (kW/sf)', fontsize=20)
    comstock_graph.tick_params(axis='y', labelsize=17)

    # Legend information and showing graph for Marimo app
    ami_data2, ami_labels2 = ami2.get_legend_handles_labels()
    comstockgraph, comstock_label = comstock_graph.get_legend_handles_labels()
    ami2.legend(ami_data2 + comstockgraph, ami_labels2 + comstock_label, fontsize=25)
    plt.gca()
    return (
        ami2,
        ami_avg_day_2,
        ami_data2,
        ami_format_2,
        ami_labels2,
        comstock_graph,
        comstock_label,
        comstockgraph,
        fig2,
        group_2,
        hour_str_2,
        hours_2,
        nhec_ami_kmeans_2,
        res_group_2,
        res_plot_comstock_ex,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        - When comparing NHEC AMI data to the **Resstock** GISMo averages, the statistics are:
            - RMSE for the GISMo forecast compared to the k-means 2 group data is 0.00142
            - NRMSE divided by the standard deviation for 2 k-mean groups is 1.58
            - NRMSE divided by the mean for 2 k-mean groups is 0.707
        """
    )
    return


@app.cell
def __(ami_avg_day_2, avg_comstock_norm, avg_gismo_norm, find_rmse, np):
    rmse_ami_2 = find_rmse(ami_avg_day_2.values, avg_comstock_norm)
    rmse_ami_2r = find_rmse(ami_avg_day_2.values, avg_gismo_norm)

    rsme_std_g2_ami = rmse_ami_2 / (np.std(ami_avg_day_2.values))
    rsme_std_g2_gismo = rmse_ami_2 / (np.std(avg_comstock_norm))

    print(f'The RMSE for the GISMo forecast compared to the k-means 2 group data is {rmse_ami_2}.')
    print("")
    print(f'The AMI RMSE divided by standard deviation for 2 k-mean groups is {rsme_std_g2_ami}.')
    print(f'The GISMo RMSE divided by standard deviation for 2 k-mean groups is {rsme_std_g2_gismo}.')
    print("")
    print(f'The coefficient of variation for Comstock and NHEC is {rmse_ami_2 / (np.mean(ami_avg_day_2.values))}')
    return rmse_ami_2, rmse_ami_2r, rsme_std_g2_ami, rsme_std_g2_gismo


if __name__ == "__main__":
    app.run()
