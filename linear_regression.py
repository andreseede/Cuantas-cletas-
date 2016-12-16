import pandas as pd
df = pd.read_csv('train.csv')
df.columns = ['instant','dteday','season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt']
print df.head()
# Inside df.head() put the number of rows you want to be shown !