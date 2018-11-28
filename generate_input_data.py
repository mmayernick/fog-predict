# This script generates input data from three 
# NOAA weather sources in the San Francisco Bay area:
# Hayward, Livermore, and San Jose.
#
# Included features are, in order: date, average wind
# speed, high temperature, low temperature, wind 
# direction (F2), and wind speed (F2).
#
# INPUT: Weather station CSV data file
# RETURN: Three location specific CSV files

import csv

# Setup input file and three output files
# for each location.
raw_data = 'data/1550964.csv'
hayward_data = 'data/hayward.csv'
livermore_data = 'data/livermore.csv'
sanjose_data = 'data/sanjose.csv'

# Open four files for reading / writing
with open(raw_data) as data_file, open(livermore_data, 'w+') as livermore, \
open(sanjose_data, 'w+') as sanjose, open(hayward_data, 'w+') as hayward:

  # Initalize CSV reader and writers
  csv_reader = csv.reader(data_file, delimiter=',')
  hw_writer = csv.writer(hayward, delimiter=',')
  lm_writer = csv.writer(livermore, delimiter=',')
  sj_writer = csv.writer(sanjose, delimiter=',')

  # Set column headers for three output files
  col_headers = ['DATE', 'AWND', 'TMAX', 'TMIN', 'WDF2', 'WSF2']
  hw_writer.writerow(col_headers)
  lm_writer.writerow(col_headers)
  sj_writer.writerow(col_headers)

  # Loop through each row in the input CSV file
  # and write to respective location file based 
  # on station ID number.
  for row in csv_reader:
      # Hayward
      if row[0] == 'USW00093228':
        # Exclude uncomplete rows
        if row[7] != '':
          # Write row with selected columns
          hw_writer.writerow([row[2], row[3], row[4], row[5], row[6].strip(), row[7]])
      # Livermore
      elif row[0] == 'USW00023285':
        if row[7] != '':
          lm_writer.writerow([row[2], row[3], row[4], row[5], row[6].strip(), row[7]])
      # San Jose
      elif row[0] == 'USW00023293':
        if row[7] != '':
          sj_writer.writerow([row[2], row[3], row[4], row[5], row[6].strip(), row[7]])