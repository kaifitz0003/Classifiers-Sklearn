from sklearn.datasets import load_wine
import seaborn as sns
import matplotlib.pyplot as plt

# We load the data as a SkLearn bunch. The items "data" and "frame" are both Pandas DataFrames. 
# The only difference between "data" and "frame" is that "frame" has an extra column called "target". 
# "target" can be used for the color because it has the labels/classes
bunch_wine = load_wine(as_frame = True) 
df = bunch_wine.frame 

sns.pairplot(df, corner = True, hue = 'target') # Plot the data
plt.savefig('EDA_Wine_pairplot.png', bbox_inches="tight") # Save the plot

# Flavanoids and proline are the most clear columns 
sns.scatterplot(data=df, x='flavanoids', y='proline', hue = 'target')

# Tells things about the data
print(df.info()) # Info on each column
print(df.describe()) # Statisical info on each numerical column
print(df['target'].value_counts()) # Tells how many times each class is entered
