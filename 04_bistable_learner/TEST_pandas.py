import pandas as pd 
from itertools import product

df = pd.DataFrame(0, index=['d', 'b', 'a'], columns=['two', 'three'])

print(df)

df.loc['d','two'] = [1]
df.loc['b','two'] = 2
df.loc['a','two'] = 3
df.loc['d','three'] = 4
df.loc['b','three'] = 5
df.loc['a','three'] = 6


print(df)


df = 3 * df ** 2

print(df)

for a, b in product(df.index, df.columns):
    print(df.loc[a, b])