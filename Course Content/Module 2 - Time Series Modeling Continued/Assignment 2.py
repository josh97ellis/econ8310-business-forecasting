from pygam import LinearGAM, s, f, l
import pandas as pd
import patsy as pt
import numpy as np

# Prepare X and Y data
data = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/Assignment2/assignment2.csv')
eqn = "trips ~ -1 + year + month + day + hour"
y, x = pt.dmatrices(eqn, data=data)

# Initialize and fit model
model = LinearGAM(s(0) + l(1) + s(2) + s(3))
modelFit = model.gridsearch(np.asarray(x), y)

# Make predictions
test_data = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/Assignment2/assignment2test.csv')
test_data = test_data.drop(columns=['Timestamp'])
pred = modelFit.predict(test_data)

print(pred)