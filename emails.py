import pandas as pd
from sklearn import preprocessing
import numpy as np

df = pd.read_csv('cleaned_data.csv')

email = []
for i in df['custAttr2']:
    num = i.find('@')
    str = i[num+1:]
    email.append(str)

df['custAttr2'] = email

lbl = preprocessing.LabelEncoder()

lbl.fit(np.unique(list(df['custAttr2'].values)))
df['custAttr2'] = lbl.transform(list(df['custAttr2'].values))

df['email_risk'] = df['custAttr2']

df = df.drop(['custAttr2'], 1)

print(df.head())

df.to_csv('cleaned_data2.csv')
