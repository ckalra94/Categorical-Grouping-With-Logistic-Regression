# LogisticGrouping

Purpose: To discover which cohorts (groupings of categorical variables) are the most likely to have their binary dependent variables = YES.

Requirements: 1 .csv with the binary DV in the first column, and at least 2 categorical variables. Continuous variables will be used for modeling, but are not required.
Import the data in LINE 47: "FakeData=PATH_TO_YOUR_CSV"

Optional: An additional .csv file with the same columns, excluding the DV, but including all other columns. Predictions for these rows will be made using the model obtained from the required .csv
The optional data is imported at LINE 129: "NewData=PATH_TO_YOUR_CSV", and change LINE 117 to "has_binary='Yes'"
