import seaborn as sns
import matplotlib.pyplot as plt

sns.get_dataset_names()
df=sns.load_dataset("diamonds")

# Pairplot of all 7 numerical columns, includes carat, clarity, depth, table, price, x, y, z, 7x7 pair plot
#sns.pairplot(df, corner = True, hue='color')

# Limited pairplot, includes carat, depth, table and price, 4x4 pair plot
df2=df.loc[:,'carat':'price']
sns.pairplot(data=df2, corner=True, hue='cut')

# Plots 4 columns in a 2d plot
plt.figure()# Creates a seperate figure for the scatter plot below
sns.scatterplot(data=df, x="carat", y="price",  size='depth', hue="cut")
