import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Replace 'cpu.csv' with the correct path to your CSV file
df = pd.read_csv('cpu.csv', header=None)

# Assuming the data format is just numeric values without headers
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.plot(np.arange(0, len(df.iloc[:, 0])), df.iloc[:, 0], marker='o', linestyle='-', color='b', label='Data')
plt.title('Plot of Column1 vs Column2')  # Add a title
plt.xlabel('Time(s)')  # Label for x-axis
plt.ylabel('CPU Percentage')  # Label for y-axis
plt.grid(True)  # Show grid
plt.legend()  # Show legend if applicable
plt.tight_layout()  # Ensure tight layout
plt.show()
