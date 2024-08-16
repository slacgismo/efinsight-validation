# efinsight-validation
## Data validation of national electrification study for residential and commercial sectors (SULI 2024)

This repository works on validating the simulation data from the North American Electrification Loadshape Forecasting tool (https://marimo.io/@gismo/na-electrification-loadshape-forecasting) with AMI data from Orange County, California and Concord, New Hampshire.

The California AMI data is available in the `sce_ami` folder, which is the only large file in this repo; this data was provided by Southern California Edison. The New Hampshire data is available in the `NHEC_Files` folder, and was provided by Power System Engineering, Inc from the NHEC Feeder 14. These data sets have been organized by `organize_data.py`. The output for the NHEC data is in `Loadshape_files`, but the output for the SCE data must be acquired from `organize_data.py`, since the edited SCE data exceeds the limit of GitHub files. Metadata from the NREL ResStock and ComStock databases are available in `metadata`, which also has house sqaure footage data for Orange County. Normalized power data from the NREL ResStock and ComStock databases are also available in `comstock_normalized_power` and `resstock_normalized_power`. The Loadshape pipeline (https://github.com/openfido/loadshape) was used to sort the AMI data, and the outputs used in the final analysis of this project are located in `Loadshape_files`. To run the Loadshape pipeline, please download the `config.csv` file located in `Loadshape_files`. `organize_data.py` is the main file that runs these files to run the validation process.

# organize_data.py

The first section of the file explains the functions of the files and gives their exact variables. The second and third sections organize the AMI data into .csv files, and includes timestamps, power values, daylight savings, and any UTC offsets. These .csv files can be used in the Loadshape pipeline, whic requires the user to pick the number of groups to sort the data into. The fourth section organizes the data from the output file, `loadshapes.csv`, and finds the mean of the standard deviations of the data to find the best group number for the data in `loadshapes.csv`. The fifth section sorts the data from the Loadshape Forecasting tool and uses the metadata to find the average loadshape for a day and a single building. Marimo features to create interative graphs via a dropdown menu to improve the visibility of ResStock and ComStock data. The sixth section finds the average square footage for a house in Orange County and New Hampshire. The final section finds the root mean squared error and the normalized root mean square errors of the AMI data and the simulation data.

## Using the Loadshape pipeline

- The Loadshape pipeline requires an ID column, data column, timezone column, and a datetime column.
    - This pipeline groups houses with similar load shapes. When using residential data, the group with the biggest count was assumed to be the group containing residential data
    - For UTC, since the offset is already included in the timestamp column, the timezone column has 0s.
    - For local time, daylight savings is recorded with a 1, and 0 refers to standard time.
- Installing the Loadshape pipeline on a Mac:
    - Activate your environment and changed the directory to the folder that has your config.csv and data file.
        - If you do not have a config.csv file, Loadshape will create one in the selected folder during installation
        - An example config.csv file is also available in `Loadshape_files`
    - Activate Arras Energy through Docker and use the command "docker run -itv $PWD:/app lfenergy/arras:develop bash"
        - This step can be optional in some cases
    - Install the pipeline with "openfido install loadshape"
    - In order for the pipeline to read your file, input "export OPENFIDO_INPUT=/app" and "export OPENFIDO_OUTPUT=/app"
    - In the config.csv file, input columns using their column names. Input your data file using its file name
        - Make sure to give "OUTPUT_PNG" a file name so it will output; it allows the user determine the residential group
        - The rest can stay the same, unless your datetime format is different
    - Group count is up to the user, but this Marimo document offers code to find the best group count for this pipeline
    - Run the pipeline using "openfido run loadshape"
    - Outputs
        - groups.csv: tells user the group each ID was assigned to
        - loadshapes.csv: tells the user the means for each seasonal, daytype, hour combination (192 combos) for each group
        - output.png: graphs all the means in loadshapes.csv for each group
 - For this project, the residential group was assumed to be the group with the biggest sample number (N) and closest to the expected power values of a residential loadshape (1-2 kW).

# Future Work
- Take a look at ORNL simulation output data: https://zenodo.org/records/10419620
   - Has the following data fields: Electricity_Facility[kBTU], NaturalGas_Facility[kBTU], Heating_Electricity[kBTU], Cooling_Electricity[kBTU], Heating_NaturalGas[kBTU], Heating_Total[kBTU], WaterSystems_Electricity[kBTU], Lighting_Electricity[kBTU], Equipment_Electricity[kBTU], Fans_Electricity[kBTU], Pumps_Electricity[kBTU], HeatRejection_Electricity[kBTU], HeatRecovery_Electricity[kBTU], Surface_Outside_Face_Heat_Emission[GJ], Zone_Exfiltration_Heat_Loss[GJ], Zone_Exhaust_Air_Heat_Loss[GJ], Heat_Rejection_Energy[GJ], Anthropogenic_Emissions[GJ]
   - ORNL also provided this sample, with archetypes: https://zenodo.org/records/7714167#.ZC3Cy3bML-g
- Look at validation process used by NREL for their ResStock and ComStock databases
