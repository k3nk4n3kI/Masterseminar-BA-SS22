#-----------Imports-----------
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

#Two Lists because after 7400 Requestes system crashed and after restart new lists were created


#-----------Lists1-----------

Preise=[]
Preise_Sug=[]
Extras_eng=[]
Colours=[]
Groups=[]
Collections=[]
lks=[]
Modles=[]
msvs=[]
Nametags=[]
Rarity=[]
Stars=[]
Status0=[]
Status1=[]
Status2=[]
Status3=[]
Title0=[]
Title1=[]
Title2=[]
Title3=[]
Trades=[]
Usages=[]
votings=[]
Conditions=[]



#-----------Lists2-----------

Prices=[]
Prices_Sug=[]
Extras=[]
Farben=[]
Gruppe=[]
Kollektionen=[]
LKs=[]
Modelle=[]
MSVs=[]
Namenschilder=[]
Seltenheiten=[]
Sterne=[]
Stati0=[]
Stati1=[]
Stati2=[]
Stati3=[]
Titels0=[]
Titels1=[]
Titels2=[]
Titels3=[]
Handels=[]
Abnutzungen=[]
Votings=[]
Zustände=[]

#-----------Lists final-----------

Prices_fin=[]
Prices_Sug_fin=[]
Extras_eng_fin=[]
Colours_fin=[]
Groups_fin=[]
Collections_fin=[]
lks_fin=[]
Modles_fin=[]
msvs_fin=[]
Nametags_fin=[]
Raririty_fin=[]
Stars_fin=[]
Status0_fin=[]
Status1_fin=[]
Status2_fin=[]
Status3_fin=[]
Title0_fin=[]
Title1_fin=[]
Title2_fin=[]
Title3_fin=[]
Trades_fin=[]
Usages_fin=[]
votings_fin=[]
Conditions_fin=[]


#-----------Loading Lists1-----------

#Prices

preis = open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/price.txt", "r").read()
preis = preis.replace("'", '').strip("[]")
preis = preis.split(' ')

for p in preis:
    p = p.strip(",").replace('.', '').replace(',', '.')
    p = float(p)
    Preise.append(p)
    

    
#Prices_Sug

preis_sug = open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/price_sug.txt", "r").read()
preis_sug = preis_sug.replace("'", '').strip("[]")
preis_sug = preis_sug.split(' ')

for p in preis_sug:
    p = p.strip(",").replace('.', '').replace(',', '.')
    p = float(p)
    Preise_Sug.append(p)
    

#Extras

extras_eng=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Extras.txt", "r").read()
extras_eng = extras_eng.replace("'", '').strip("[]")
extras_eng = extras_eng.split(" ")

for e in extras_eng:
    e = e.strip(",").replace('★', 'StatTrak™')
    if e != "StatTrak™":
        if e != "Souvenir":
            e = "0"
    Extras_eng.append(e)

#Farbe

colour = open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Farbverlauf.txt", "r").read()
colour = colour.replace("'", '').strip("[]")
colour = colour.split(' ')

for f in colour:
    f = f.strip(",")
    Colours.append(int(f))
    
#Gruppe

group=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/group.txt", "r").read()
group = group.replace("'", '').strip("[]")
group = group.split(' ')


for g in group:
    g = g.strip(",")
    Groups.append(g)


#Kollektion

collection=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/kollektion.txt", "r").read()
collection = collection.replace("'", '').strip("[]")
collection = collection.split(',')


for k in collection:
    k = k.strip(',').strip()
    Collections.append(k)

#LK

lk=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/LK.txt", "r").read()
lk = lk.replace("'", '').strip("[]")
lk = lk.split(' ')

for l in lk:
    l = l.strip(',')
    lks.append(int(l))

#Modell

modle=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/modell.txt", "r").read()
modle = modle.replace("'", '').strip("[]")
modle = modle.split(',')

for m in modle:
    m = m.strip()
    Modles.append(m)

#MSV

msv=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/MSV.txt", "r").read()
msv = msv.replace("'", '').strip("[]")
msv = msv.split(' ')

for m in msv:
    m=m.strip(',')
    msvs.append(int(m))

#Namensschild

names=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Namensschild.txt", "r").read()
names = names.replace("'", '').strip("[]")
names = names.split(' ')

for n in names:
    n = n.strip(',')
    Nametags.append(int(n))

#Seltenheit

rare=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/rarity.txt", "r").read()
rare = rare.replace("'", '').strip("[]")
rare = rare.split(',')

for s in rare:
    s = s.strip()
    Rarity.append(s)

#Stern

star=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/stars.txt", "r").read()
star = star.replace("'", '').strip("[]")
star = star.split(' ')

for s in star:
    s=s.strip(',')
    Sterne.append(float(s))

#Status0

status0=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Sticker_Status0.txt", "r").read()
status0 = status0.replace("'", '').strip("[]")
status0 = status0.split(',') #remove %

for s in status0:
    s = s.strip()
    if 'Verkratzt ' in s:
        s = 'Verkratzt'
        Status0.append(s)
    elif s == 'None':
        s = 'Kein Sticker'
        Status0.append(s)
    else:
        Status0.append(s)

#Status1

status1=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Sticker_Status1.txt", "r").read()
status1 = status1.replace("'", '').strip("[]")
status1 = status1.split(',')

for s in status1:
    s = s.strip()
    if 'Verkratzt ' in s:
        s = 'Verkratzt'
        Status1.append(s)
    elif s == 'None':
        s = 'Kein Sticker'
        Status1.append(s)
    else:
        Status1.append(s)


#Status2

status2=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Sticker_Status2.txt", "r").read()
status2 = status2.replace("'", '').strip("[]")
status2 = status2.split(',')

for s in status2:
    s = s.strip()
    if 'Verkratzt ' in s:
        s = 'Verkratzt'
        Status2.append(s)
    elif s == 'None':
        s = 'Kein Sticker'
        Status2.append(s)
    else:
        Status2.append(s)


#Staus3

status3=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Sticker_Status3.txt", "r").read()
status3 = status3.replace("'", '').strip("[]")
status3 = status3.split(',')

for s in status3:
    s = s.strip()
    if 'Verkratzt ' in s:
        s = 'Verkratzt'
        Status3.append(s)
    elif s == 'None':
        s = 'Kein Sticker'
        Status3.append(s)
    else:
        Status3.append(s)


#Titel0

title0=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Sticker_Titel0.txt", "r").read()
title0 = title0.replace("'", '').strip("[]")
title0 = title0.split(' ')

for t in title0:
    t = t.strip(',')
    Title0.append(int(t))


#Titel1

title1=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Sticker_Title1.txt", "r").read()
title1 = title1.replace("'", '').strip("[]")
title1 = title1.split(' ')

for t in title1:
    t = t.strip(',')
    Title1.append(int(t))

#Titel2

title2=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Sticker_Title2.txt", "r").read()
title2 = title2.replace("'", '').strip("[]")
title2 = title2.split(' ')

for t in title2:
    t = t.strip(',')
    Title2.append(int(t))

#Titel3

title3=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Sticker_Title3.txt", "r").read()
title3 = title3.replace("'", '').strip("[]")
title3 = title3.split(' ')

for t in title3:
    t = t.strip(',')
    Title3.append(int(t))

#Handel

trade=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Trade.txt", "r").read()
trade = trade.replace("'", '').strip("[]")
trade = trade.split(' ')

for h in trade:
    h = h.strip(',')
    Trades.append(float(h))

#Abnutzung

use=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/Usage.txt", "r").read()
use = use.replace("'", '').strip("[]")
use = use.split(' ')

for a in use:
    a = a.strip(',')
    Usages.append(float(a))

#Voting

vot=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/vot.txt", "r").read()
vot = vot.replace("'", '').strip("[]")
vot = vot.split(' ')

count = 0
for v in vot:
    v = v.strip(',')
    count +=1
    if count <= 7287:
        v = float(v)
        v = v * 10
        v = int(v)
        votings.append(v)
    else:
        votings.append(int(v))

#Zustand

condition=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/zustand.txt", "r").read()
condition = condition.replace("'", '').strip("[]")
condition = condition.split(' ')

for z in condition:
    z = z.strip(',')
    Conditions.append(z)



#-----------Loading Lists2-----------

#Prices

price = open("/Users/lars/Documents/Uni/VSCodeTest/price2.txt", "r").read()
price = price.replace("'", '').strip("[]")
price = price.split(' ')

for p in price:
    p = p.strip(",").replace('.', '').replace(',', '.')
    p = float(p)
    Prices.append(p)
    
#Prices_Sug

price_sug = open("/Users/lars/Documents/Uni/VSCodeTest/price_sug2.txt", "r").read()
price_sug = price_sug.replace("'", '').strip("[]")
price_sug = price_sug.split(' ')

for p in price_sug:
    p = p.strip(",").replace('.', '').replace(',', '.')
    p = float(p)
    Prices_Sug.append(p)
    
ps = {"Preis_Sug": Prices_Sug}
df_ps = pd.DataFrame(ps)
df_ps['Preis_Sug'].unique()

#Extras

extras=open("/Users/lars/Documents/Uni/VSCodeTest/Extras2.txt", "r").read()
extras = extras.replace("'", '').strip("[]")
extras = extras.split(" ")

for e in extras:
    e = e.strip(",").replace('★', 'StatTrak™')
    if e != "StatTrak™":
        if e != "Souvenir":
            e = "0"
    Extras.append(e)

#Farbe

farbe = open("/Users/lars/Documents/Uni/VSCodeTest/Farbverlauf2.txt", "r").read()
farbe = farbe.replace("'", '').strip("[]")
farbe = farbe.split(' ')

for f in farbe:
    f = f.strip(",")
    Farben.append(int(f))
    
#Gruppe

gruppe=open("/Users/lars/Documents/Uni/VSCodeTest/group2.txt", "r").read()
gruppe = gruppe.replace("'", '').strip("[]")
gruppe = gruppe.split(' ')


for g in gruppe:
    g = g.strip(",")
    Gruppe.append(g)


#Kollektion

kollektion=open("/Users/lars/Documents/Uni/VSCodeTest/kollektion2.txt", "r").read()
kollektion = kollektion.replace("'", '').strip("[]")
kollektion = kollektion.split(',')


for k in kollektion:
    k = k.strip(',').strip()
    Kollektionen.append(k)

#LK

lk=open("/Users/lars/Documents/Uni/VSCodeTest/LK2.txt", "r").read()
lk = lk.replace("'", '').strip("[]")
lk = lk.split(' ')

for l in lk:
    l = l.strip(',')
    LKs.append(int(l))

#Modell

modell=open("/Users/lars/Documents/Uni/VSCodeTest/modell2.txt", "r").read()
modell = modell.replace("'", '').strip("[]")
modell = modell.split(',')

for m in modell:
    m = m.strip()
    Modelle.append(m)

#MSV

msv=open("/Users/lars/Documents/Uni/VSCodeTest/MSV2.txt", "r").read()
msv = msv.replace("'", '').strip("[]")
msv = msv.split(' ')

for m in msv:
    m=m.strip(',')
    MSVs.append(int(m))

#Namensschild

name=open("/Users/lars/Documents/Uni/VSCodeTest/Namensschild2.txt", "r").read()
name = name.replace("'", '').strip("[]")
name = name.split(' ')

for n in name:
    n = n.strip(',')
    Namenschilder.append(int(n))

#Seltenheit

selten=open("/Users/lars/Documents/Uni/VSCodeTest/rarity2.txt", "r").read()
selten = selten.replace("'", '').strip("[]")
selten = selten.split(',')

for s in selten:
    s = s.strip()
    Seltenheiten.append(s)

#Stern

stern=open("/Users/lars/Documents/Uni/VSCodeTest/stars2.txt", "r").read()
stern = stern.replace("'", '').strip("[]")
stern = stern.split(' ')

for s in stern:
    s=s.strip(',')
    Sterne.append(float(s))

#Status0

status0=open("//Users/lars/Documents/Uni/VSCodeTest/Sticker_Status02.txt", "r").read()
status0 = status0.replace("'", '').strip("[]")
status0 = status0.split(',') #remove %

for s in status0:
    s = s.strip()
    if 'Verkratzt ' in s:
        s = 'Verkratzt'
        Stati0.append(s)
    elif s == 'None':
        s = 'Kein Sticker'
        Stati0.append(s)
    else:
        Stati0.append(s)

#Status1

status1=open("/Users/lars/Documents/Uni/VSCodeTest/Sticker_Status12.txt", "r").read()
status1 = status1.replace("'", '').strip("[]")
status1 = status1.split(',')

for s in status1:
    s = s.strip()
    if 'Verkratzt ' in s:
        s = 'Verkratzt'
        Stati1.append(s)
    elif s == 'None':
        s = 'Kein Sticker'
        Stati1.append(s)
    else:
        Stati1.append(s)


#Status2

status2=open("/Users/lars/Documents/Uni/VSCodeTest/Sticker_Status22.txt", "r").read()
status2 = status2.replace("'", '').strip("[]")
status2 = status2.split(',')

for s in status2:
    s = s.strip()
    if 'Verkratzt ' in s:
        s = 'Verkratzt'
        Stati2.append(s)
    elif s == 'None':
        s = 'Kein Sticker'
        Stati2.append(s)
    else:
        Stati2.append(s)


#Staus3

status3=open("/Users/lars/Documents/Uni/VSCodeTest/Sticker_Status32.txt", "r").read()
status3 = status3.replace("'", '').strip("[]")
status3 = status3.split(',')

for s in status3:
    s = s.strip()
    if 'Verkratzt ' in s:
        s = 'Verkratzt'
        Stati3.append(s)
    elif s == 'None':
        s = 'Kein Sticker'
        Stati3.append(s)
    else:
        Stati3.append(s)


#Titel0

titel0=open("/Users/lars/Documents/Uni/VSCodeTest/Sticker_Titel02.txt", "r").read()
titel0 = titel0.replace("'", '').strip("[]")
titel0 = titel0.split(' ')

for t in titel0:
    t = t.strip(',')
    Titels0.append(int(t))


#Titel1

titel1=open("/Users/lars/Documents/Uni/VSCodeTest/Sticker_Title12.txt", "r").read()
titel1 = titel1.replace("'", '').strip("[]")
titel1 = titel1.split(' ')

for t in titel1:
    t = t.strip(',')
    Titels1.append(int(t))

#Titel2

titel2=open("/Users/lars/Documents/Uni/VSCodeTest/Sticker_Title22.txt", "r").read()
titel2 = titel2.replace("'", '').strip("[]")
titel2 = titel2.split(' ')

for t in titel2:
    t = t.strip(',')
    Titels2.append(int(t))

#Titel3

titel3=open("/Users/lars/Documents/Uni/VSCodeTest/Sticker_Title32.txt", "r").read()
titel3 = titel3.replace("'", '').strip("[]")
titel3 = titel3.split(' ')

for t in titel3:
    t = t.strip(',')
    Titels3.append(int(t))

#Handel

handel=open("/Users/lars/Documents/Uni/VSCodeTest/Trade2.txt", "r").read()
handel = handel.replace("'", '').strip("[]")
handel = handel.split(' ')

for h in handel:
    h = h.strip(',')
    Handels.append(float(h))

#Abnutzung

abnutzung=open("/Users/lars/Documents/Uni/VSCodeTest/Usage2.txt", "r").read()
abnutzung = abnutzung.replace("'", '').strip("[]")
abnutzung = abnutzung.split(' ')

for a in abnutzung:
    a = a.strip(',')
    Abnutzungen.append(float(a))

#Voting

voting=open("/Users/lars/Documents/Uni/VSCodeTest/vot2.txt", "r").read()
voting = voting.replace("'", '').strip("[]")
voting = voting.split(' ')

count = 0
for v in voting:
    v = v.strip(',')
    count +=1
    if count <= 7287:
        v = float(v)
        v = v * 10
        v = int(v)
        Votings.append(v)
    else:
        Votings.append(int(v))

#Zustand

zustand=open("/Users/lars/Documents/Uni/VSCodeTest/zustand2.txt", "r").read()
zustand = zustand.replace("'", '').strip("[]")
zustand = zustand.split(' ')

for z in zustand:
    z = z.strip(',')
    Zustände.append(z)
    

#-----------Merging Lists-----------

Prices_fin = Preise + Prices
Prices_Sug_fin = Preise_Sug + Prices_Sug
Extras_eng_fin = Extras_eng + Extras
Colours_fin = Colours + Farben
Groups_fin = Groups + Gruppe
Collections_fin = Collections + Kollektionen
lks_fin = lks + LKs
Modles_fin = Modles + Modelle
msvs_fin = msvs + MSVs
Nametags_fin = Nametags + Namenschilder
Raririty_fin = Rarity + Seltenheiten
Stars_fin = Stars + Sterne
Status0_fin = Status0 + Stati0
Status1_fin = Status1 + Stati1
Status2_fin = Status2 + Stati2
Status3_fin = Status3 + Stati3
Title0_fin = Title0 + Titels0
Title1_fin = Title1 + Titels1
Title2_fin = Title2 + Titels2
Title3_fin = Title3 + Titels3
Trades_fin = Trades + Handels
Usages_fin = Usages + Abnutzungen
votings_fin = votings + Votings
Conditions_fin = Conditions +  Zustände

#-----------Dataframe-----------

daten_ges = {'Gruppe':Groups_fin, 'Modell':Modles_fin, 'Kollektion':Collections_fin, 
        'Seltenheit':Raririty_fin, 'Zustand':Conditions_fin, 'Abnutzung':Usages_fin, 
        'Extras':Extras_eng_fin, 'Namensschild':Nametags_fin, 'Handel':Trades_fin, 
        'Farbe':Colours_fin, 'Lack':lks_fin, 'MSV':msvs_fin, 'Sterne':Stars_fin, 'Voting':votings_fin, 
        'Status0':Status0_fin, 'Status1':Status1_fin, 'Status2':Status2_fin, 'Status3':Status3_fin, 
        'Titel0':Title0_fin, 'Titel1':Title1_fin, 'Titel2':Title2_fin, 'Titel3':Title3_fin,
        'Preis':Prices_fin, 'Preis_Sug':Prices_Sug_fin}

df_scraper_ges = pd.DataFrame(daten_ges)

#df_scraper_ges.to_csv("/Users/lars/Documents/Uni/VSCodeTest/Daten_Scraper.csv")


#-----------Adding Discount-----------

def Discount(col1, col2):
    return ((col2 - col1) / col1) * 100
    

df_scraper_ges['Rabatt'] = Discount(df_scraper_ges['Preis'], df_scraper_ges['Preis_Sug'])   

df_scraper_ges.loc[df_scraper_ges['Rabatt'] < 0, 'Rabatt'] = 0
df_scraper_ges['Rabatt'] = df_scraper_ges['Rabatt'].astype('int')

df_scraper_ges.to_csv("/Users/lars/Documents/Uni/VSCodeTest/Daten_Scraper.csv")


#-----------Preprocessing-----------

#siehe: https://medium.com/swlh/all-you-need-to-know-about-handling-high-dimensional-data-7197b701244d

#-----------Columns with similar Values-----------

df_scraper = df_scraper_ges

def get_similar_values(df, percent=90):
    count = 0
    sim_val_col=[]
    for col in df.columns:
        percent_vals = (df[col].value_counts()/len(df)*100).values
        if percent_vals[0] > percent and len(percent_vals) > 2:
            sim_val_col.append(col)
            count +=0
    print('Total Columns with majority singular values share: ', count)
    return sim_val_col



get_similar_values(df_scraper)

df_scraper = df_scraper.drop(['Handel', 'Farbe', 'Status3'], axis=1) #replaced both dfs

#-----------One-Hot-Encoding-----------

columns = ['Gruppe', 'Modell', 'Kollektion', 'Seltenheit', 'Zustand', 'Extras', 'Status0', 'Status1', 'Status2']

for col in columns:
    df_scraper = pd.concat([df_scraper.drop([col], axis=1),
                   pd.get_dummies(df_scraper[col],
                    drop_first=True, prefix=col)], axis=1)
    

df_scraper.to_csv("/Users/lars/Documents/Uni/Seminar_BA/Daten_Scraper.csv")

#-----------Normalization-----------

#not necessary: https://stats.stackexchange.com/questions/353462/what-are-the-implications-of-scaling-the-features-to-xgboost


#-----------Removing Outliers-----------

columns = ['Abnutzung', 'Lack', 'MSV', 'Sterne', 'Voting', 'Preis', 'Preis_Sug']

"""
:param df: input data in the form of a dataframe
:param columns: columns where the outliers need to be found
:return: df_clean: cleaned dataframe where the outliers have been removed
"""

# Select the indices for data points you wish to remove
outliers_lst = []
leave_cols = []  # columns you may want to leave out
# For each feature find the data points with extreme high or low values
for feature in columns:

    if feature not in leave_cols:
        Q1 = df_scraper[feature].quantile(0.25)
        Q3 = df_scraper[feature].quantile(0.75)
        step = 1.5 * (Q3 - Q1)

        # Display the outliers
        # print("Data points considered outliers for the feature '{}':".format(feature))

        # finding any points outside of Q1 - step and Q3 + step
        outliers_rows = df_scraper.loc[~((df_scraper[feature] >= Q1 - step) & (df_scraper[feature] <= Q3 + step)), :]
        # display(outliers_rows)

        outliers_lst.append(list(outliers_rows.index))

outliers = list(itertools.chain.from_iterable(outliers_lst))
# List of duplicate outliers
dup_outliers = list(set([x for x in tqdm(outliers, position=0) if outliers.count(x) > 1]))
df_clean = df_scraper.loc[~df_scraper.index.isin(dup_outliers)]
print("Processed outliers")


df_clean.to_csv("/Users/lars/Documents/Uni/Seminar_BA/Daten_Scraper_Clean.csv")

df_clean = pd.read_csv('/Users/lars/Documents/Uni/Seminar_BA/Daten_Scraper_Clean.csv')


#-----------PCA-----------

#siehe: https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
#siehe: https://towardsdatascience.com/how-to-use-pca-tsne-xgboost-and-finally-bayesian-optimization-to-predict-the-price-of-houses-626dbaf242ae

df_scraper_pca = df_clean.copy()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

X1_pca = df_scraper_pca.drop(['Preis', 'Preis_Sug'], axis=1)
y1_pca = df_scraper_pca['Preis']

#-----------Scaling Data-----------
scaler = StandardScaler()

X_pca_fit = scaler.fit_transform(X1_pca)

#-----------PCA for 95%-----------

pca = PCA()

pca.fit_transform(X_pca_fit)
df_pca = pca.transform(X_pca_fit)

#-----------Getting most important feature of each component-----------

n_pcs = pca.components_.shape[0]

most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names=X1_pca.columns.values.tolist()

most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df = pd.DataFrame(sorted(dic.items()))

cum_var = np.cumsum(pca.explained_variance_ratio_)


print(len( pca.components_ ))

#-----------Visulazing complette varianz-----------

plot.plot(cum_var, label='Data')
#plot.plot(y=0.95, x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], linestyle='--', colours='red', label='0.95%')
plot.axhline(y=0.95, color='r', linestyle='--')
plot.ylabel('Cumulative Explained Variance')
plot.xlabel('Number of Components')
plot.show()


#-----------Calculating and Visulazing most important Feat Groups-----------

pca_feat_list = df[1].tolist()

pca_individ_feat = []

for i in pca_feat_list:
    if i in pca_individ_feat:
        continue
    else:
        pca_individ_feat.append(i)
print(len(pca_individ_feat))

pca_feat_list.count('Modell_Karambit')

pca_individ_feat_amount=[]

for i in df[1].unique().tolist():
     amount = pca_feat_list.count(i)
     pca_individ_feat_amount.append(amount)

dict_pca_individ_feat = dict(zip(pca_individ_feat, pca_individ_feat_amount))

pca_cat = {'Zustand':0, 'Titel':0, 'Sterne':0, 'Seltenheit':0, 'Namensschild':0, 'Modell':0, 'Kollektion':0, 'Gruppe':0, 'Extras':0}

for k, v in dict_pca_individ_feat.items():
    if 'Zustand' in k:
        pca_cat['Zustand']+=v
    elif 'Titel' in k:
        pca_cat['Titel']+=v
    elif 'Sterne' in k:
        pca_cat['Sterne']+=v
    elif 'Seltenheit' in k:
        pca_cat['Seltenheit']+=v
    elif 'Namensschild' in k:
        pca_cat['Namensschild']+=v
    elif 'Kollektion' in k:
        pca_cat['Kollektion']+=v
    elif 'Gruppe' in k:
        pca_cat['Gruppe']+=v
    elif 'Extras' in k:
        pca_cat['Extras']+=v
    elif 'Modell' in k:
        pca_cat['Modell']+=v



categories = list(pca_cat.keys())
values = list(pca_cat.values())

plot.bar(categories, values)
plot.xticks(rotation=90)
for i in range(len(values)):
    plot.annotate(str(values[i]), xy=(categories[i],values[i]), ha='center', va='bottom')
plot.xlabel('Categories')
plot.ylabel('Amount')
plot.show()

#-----------Before vs After PCA Plot-----------

pca2 = PCA(n_components=2) # estimate only 2 PCs
X_new = pca2.fit_transform(X_pca_fit) # project the original data into the PCA space

fig, axes = plot.subplots(1,2)

#Before

axes[0].scatter(X_pca_fit[:,0], X_pca_fit[:,1], c='blue')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')

#After

axes[1].scatter(X_new[:,0], X_new[:,1], c='blue')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plot.figure(figsize=(50,50))
plot.show()







