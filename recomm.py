import pandas as pd
df_quotes=pd.read_csv("Dataset/recomm.csv")

df_user=pd.DataFrame(columns=["quote_id","rating"])
w_user=[1 for i in range(len(df_quotes.GENRE.unique()))]

# Making the one hot matrix
one_hot_cols=df_quotes.GENRE.unique()
one_hot_quotes=pd.DataFrame(data=[[0 for i in range(len(one_hot_cols))] for i in range(len(df_quotes))],columns=df_quotes.GENRE.unique())
tmp = df_quotes.GENRE 
for i in range(len(tmp)) :
    one_hot_quotes.loc[i][tmp[i]]=1

def update_weights():
    global df_user,df_quotes,w_user
    #sum of columns*ratings
    w=(one_hot_quotes.loc[df_user["quote_id"]]*w_user).sum()
    w/= w.sum()
    w_user = w.values

def recommendation(input_emotion):
    global df_quotes,df_user,w_user
    # Get recommendation dataframe
    tmp = df_quotes[(df_quotes["EMOTION"]==input_emotion)]
    tmp = tmp[~tmp.index.isin(df_user["quote_id"])]
    tmp = tmp.index #indexes of not-yet rated quotes

    # We multiply the one_hot_quotes of the recommendable quotes by the user profile (weights)
    # and see which line sums up to the biggest value
    tmp = (one_hot_quotes.loc[tmp]*w_user).sum(1).sort_values(ascending=False).index[0]
    
    return tmp #index of recommended quote