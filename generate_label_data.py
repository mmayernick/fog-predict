# This script parses weather report TXT files
# to generate a list of dates that reported fog
#
# INPUT: Monthly NOAA bouy data TXT files
# RETURN: CSV file of dates that had fog.

import csv
import glob
import os

data_directory = 'data'
fog_data = data_directory + '/fog_label.csv'
fog_labels = {}
fog_month_labels = {}

file_list = sorted(glob.glob(os.path.join(os.getcwd(), data_directory, "*.txt")))

for file_path in file_list:
  with open(file_path) as f_input:
    date = ''

    for count, line in enumerate(f_input):
      # Skip header lines 0 and 1
      if count > 1:
        # Parse line and extract date
        words = line.strip().split()
        line_date = words[0] + words[1] + words[2]

        # If line represents new date, 
        # reset fog_count counter and
        # fog label to false.
        if date != line_date:
          # Check for initalization or blank
          # lines
          if line_date != '':
            fog_count = 0      
            fog_labels[line_date] = 0

        # Fetch temperature and dew point to
        # calculate if fog forming
        temp = float(words[13])
        dewp = float(words[15])
        visb = float(words[16])
        # Validate data is present (invalid data 
        # represented by '99', '999', etc) and 
        # that fog forming.
        if temp < 99.0 and dewp < 99.0 and ((temp - 1.0) < dewp) and visb < 5.0:
          # If fog forming, add to count
          fog_count += 1
          # Check to see if fog is sustained
          # to see if count for the day is over 10
          if fog_count > 5:
            # If sustained fog present, indicated
            # there was fog on that day
            fog_labels[line_date] = 1

        # Update date tracker to
        # pick up new dates
        date = line_date

# Build count of total foggy days by month
for key, value in fog_labels.items():
  month = key[4:6]
  if month not in fog_month_labels:
    fog_month_labels[month] = 0
  if value == True:
    fog_month_labels[month] += 1

with open(fog_data, 'w+') as fog:
  fog_writer = csv.writer(fog, delimiter=',')

  for key, value in fog_labels.items():
    fog_writer.writerow([key, value])

print(fog_month_labels)