from sklearn.datasets import load_iris
from seaborn import pairplot
bunch = load_iris(as_frame=True)
df = bunch.frame

pairplot(df, corner=True, hue='target')
print(df.info())
print(df.describe())
print(df['target'].value_counts())




