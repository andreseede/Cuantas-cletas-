import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')
df.columns = ['instant','dteday','season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt']
print df.head()
# Inside df.head() put the number of rows you want to be shown !

sns.set(style='whitegrid',context='notebook')
cols = ['instant','dteday','season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt']
sns.pairplot(df[cols],size=2.5)
plt.interactive(False)
plt.show(block=True)
