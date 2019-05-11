import pandas as pd

# Load the dataset through pandas
data = pd.read_csv("../led.csv")
dataFrame = pd.DataFrame(data=data)
sortData = dataFrame.sort_values(['Led0', 'Led1', 'Led2', 'Led3', 'Led4', 'Led5', 'Led6', 'Output'])
pass
