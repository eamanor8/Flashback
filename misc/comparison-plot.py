import matplotlib.pyplot as plt
import pandas as pd
from results_data.results_4sq import data  # Importing the data from the Python file

# Convert the dictionary to DataFrame
df = pd.DataFrame(data)

# Creating a histogram for the data
plt.figure(figsize=(10, 6))

# Plotting histograms for each metric
bar_width = 0.25  # width of bars
index = range(len(df['Metric']))  # index for groups of bars

# Position of bars on x-axis
bar1 = [i - bar_width for i in index]
bar2 = index
bar3 = [i + bar_width for i in index]

plt.bar(bar1, df['Paper Results'], width=bar_width, label='Paper Results', edgecolor='gray')
plt.bar(bar2, df['4sq-Benchmark'], width=bar_width, label='4sq-Benchmark', edgecolor='gray')
plt.bar(bar3, df['Experimental'], width=bar_width, label='Experimental', color='red', edgecolor='gray')

# Adding titles and labels
plt.title('Model Performance Comparison')
plt.xlabel('Metric')
plt.ylabel('Accuracy')
plt.xticks(index, df['Metric'])
plt.legend()

# Display the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot to a file
plt.savefig('./imgs/4sq-comparison.png')
