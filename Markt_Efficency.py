#-----------Imports-----------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plot
import statistics

Skin_Name=[]

#-----------Loading Skin_Name-----------

sn=open("/Users/lars/Documents/Uni/VSCodeTest/skin_name2.txt", "r").read()
sn = sn.replace("'", '').strip("[]")
sn = sn.split(',')

sn2=open("/Users/lars/Documents/Uni/VSCodeTest/Daten_7400/skin_name.txt", "r").read()
sn2 = sn2.replace("'", '').strip("[]")
sn2 = sn2.split(',')

#-----------Loading DF-----------

df = pd.read_csv('/Users/lars/Documents/Uni/VSCodeTest/Daten_Scraper.csv')

Skin_Name = sn2 + sn



df['Skin_Name']=Skin_Name

df.to_csv("/Users/lars/Documents/Uni/VSCodeTest/Daten_Scraper_Skins.csv")

#-----------KistenPreise-----------

#Gettin Boxes and Key Names

Kisten_Namen = []

column_names =np.unique(df.columns)

Waffenkisten=[]

for i in column_names:
    if 'Modell_Waffenkiste' in i:
        Waffenkisten.append(i)


Kisten_Positionen = []      
        
for i in Waffenkisten:
    a = np.where(df[i] == 1)
    Kisten_Positionen.append(a)

Durchs_Preis=[]

for arr in Kisten_Positionen:
    Preis_Kisten = []
    for p in arr:
        Preis_Kisten = df['Preis'][p].tolist()
    Durchs_Preis.append(statistics.mean(Preis_Kisten))


    
#-----------SchlüsselPreise-----------


Schlüssel_Namen=[]

for i in column_names:
    if 'Modell_Kistenschlüssel' in i:
        Schlüssel_Namen.append(i)

#Del Schlüssel if no Box
        
Schlüssel_Namen.remove("Modell_Kistenschlüssel „Operation Phoenix“")


Schlüsse_Positionen = []      
        
for i in Schlüssel_Namen:
    a = np.where(df[i] == 1)
    Schlüsse_Positionen.append(a)

Durchs_Preis_Schlüssel=[]

for arr in Schlüsse_Positionen:
    Preis_Schlüssel = []
    for p in arr:
        # print(df['Preis'][p])
        Preis_Schlüssel = df['Preis'][p].tolist()
    Durchs_Preis_Schlüssel.append(statistics.mean(Preis_Schlüssel)) 
    
    
#-----------Creating Dict with Boxes + Keys-----------


Kollektionen = {'Chroma 3':[0.48, 32.407272727272726], 'Chroma':[0.7696652719665272, 7.0179775280898875], 'Falchion':[0.22002915451895044, 27.232], 'Gamma 2':[0.6863709677419355, 23.1657], 'Gamma':[0.6227659574468085, 54.844848484848484], 'Gefahrenzone':[0.11, 43.668124999999996], 'Handschuhe':[2.200893854748603, 51.173125],
                'Horizont':[0.24923588039867112, 41.61], 'Jagd':[4.852154471544716, 8.36357142857143], 'Operation Breakout':[2.61, 112.2475], 'Operation Hydra':[16.36391304347826, 4.106013986013986], 'Operation Vanguard':[0.8060509554140127, 10.213000000000001]}


Preis_Ges = {}

for key in Kollektionen:
    a = Kollektionen[key]
    #print(a)
    Price_Total = sum(a)
    Preis_Ges[key] = Price_Total
    print(Price_Total)
    
    
#Operation Breakout most expensive & Chroma most cheapest box



#Benötigte Infos: Welche Kollektion, welcher Seltenheit, welcher Preis

df_Erwartungswert = df[['Preis', 'Kollektion_Chroma“', 'Kollektion_Breakout“', 'Skin_Name', 'Seltenheit_Vertraulich', 'Seltenheit_Verbraucherklasse',
                       'Seltenheit_Verdeckt', 'Seltenheit_Industriequalität', 'Seltenheit_Militärstandard', 'Seltenheit_Limitiert', 'Gruppe_Messer']]

a = df.columns.tolist()

print(a)

#Odds for Skins: https://blog.lootbear.com/csgo-case-odds/

#-----------Find all ChromaItems-----------

Chroma_Positionen=[]
Chroma = np.where(df['Kollektion_Chroma“'] == 1)
Chroma_Positionen = Chroma[0].tolist()

#-----------Gettin Chroma SkinNames-----------

Skin_Name_Chroma=[]

for i in Chroma_Positionen:
    if df_Erwartungswert['Skin_Name'][i] in Skin_Name_Chroma:
        continue
    else:
        Skin_Name_Chroma.append(df['Skin_Name'][i])

#-----------Calculating MeanValue-----------

Chroma_Mean_Price = dict.fromkeys(Skin_Name_Chroma, None)

for Name in Skin_Name_Chroma:
    Name_Index = df_Erwartungswert.loc[df_Erwartungswert.Skin_Name == Name].index.tolist()
    price = []
    for num in Chroma_Positionen:
        if num in Name_Index:
            price.append(df_Erwartungswert['Preis'][num])
    print(price)
    
    average = statistics.mean(price)
    print(average)
    Chroma_Mean_Price[Name] = average
    
#-----------Missing Skins-----------
#siehe: https://csgostash.com/case/18/Operation-Breakout-Weapon-Case?Knives=1

'''
Mil-Spec (Blue) – 79,92327% = 0.7992327
Restricted (Purple) – 15,98465% = 0.1598465
Classified (Pink) – 3,19693% = 0.0319693
Covert (Red) – 0,63939% = 0.0063939
Special Items/Knives (Gold) – 0,25575% = 0.0025575

'''
#Already expected value

Chroma_Mean_Price2 ={'Marbel_Fade_Bay':2.0424578625, 'Doppler_Bay':1.1878436625, 'Tiger_Tooth_Bay':1.2978417375, 'Ultraviolett_Bay':0.6967269375, 
                     'Damascus_Bay':0.7645774124999999, 'Rust_Coat_Bay':0.5763837749999999, 'Marbel_Fade_Flip':1.0362734249999999, 'Doppler_Flip':0.8480158499999999, 'Tiger_Tooth_Flip':0.839755125,
                     'Damascus_Flip':0.5449393124999999, 'Rust_Coat_Flip':0.4155042375, 'Ultraviolett_Flip':0.6928906875, 'Marbel_Fade_Gut':0.392141475, 'Doppler_Gut':0.33480232499999996,
                     'Tiger_Tooth_Gut':0.3550705125, 'Ultraviolett_Gut':0.42849633749999994, 'Damascus_Gut':0.24457372499999996, 'Rust_Coat_Gut':0.2330777625,
                     'Marbel_Fade_Kara':3.30270435, 'Tiger_Tooth_Kara':1.9431117749999998, 'Doppler_Kara':2.262876, 'Damascus_Kara':1.5221216999999998, 'Ultraviolett_Kara':1.4613682874999998,
                     'Rust_Coat_Kara':1.3683264375, 'Marbel_Fade_M9':3.1272342749999997, 'Doppler_M9':1.9599784875, 'Tiger_Tooth_M9':2.0417417625, 'Ultraviolett_M9':0.88699215,
                     'Damascus_M9':0.9577326, 'Rust_Coat_M9':0.8431821749999999, 'Cartell':0.289322165, 'Muertos':0.1868605585, 
                    'Serenity':0.16304343000000002, 'Grotto':0.1718350305, 'Quicksilver':0.15984654, 'Naga':0.43558171}



print(0.2*0.7992327)

#-----------Calculating Expected Value for Chroma-----------

Name_Chroma=[]
for key in Chroma_Mean_Price:
    Name_Chroma.append(key)

Used_keys_Chroma=[]

Chroma_Erw={}

for pos in Chroma_Positionen:
    if df_Erwartungswert['Skin_Name'][pos] in Used_keys_Chroma:
        continue
    if df_Erwartungswert['Skin_Name'][pos] not in list(Chroma_Mean_Price.keys()):
        continue
    if df_Erwartungswert['Skin_Name'][pos] in Name_Chroma and df_Erwartungswert['Seltenheit_Verdeckt'][pos] == 1 and df_Erwartungswert['Gruppe_Messer'][pos] == 1 and df_Erwartungswert['Kollektion_Chroma“'][pos] == 1:
        print(1)
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Chroma_Mean_Price[key] * 0.0025575
        Used_keys_Chroma.append(df_Erwartungswert['Skin_Name'][pos])
        Chroma_Erw[df_Erwartungswert['Skin_Name'][pos]]=test
        
    elif df_Erwartungswert['Skin_Name'][pos] in Name_Chroma and df_Erwartungswert['Seltenheit_Vertraulich'][pos] == 1 and df_Erwartungswert['Kollektion_Chroma“'][pos] == 1:
        print(2)
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Chroma_Mean_Price[key] * 0.0319693
        Used_keys_Chroma.append(df_Erwartungswert['Skin_Name'][pos])
        Chroma_Erw[df_Erwartungswert['Skin_Name'][pos]]=test
    
    elif df_Erwartungswert['Skin_Name'][pos] in Name_Chroma and df_Erwartungswert['Seltenheit_Verdeckt'][pos] == 1 and df_Erwartungswert['Kollektion_Chroma“'][pos] == 1:
        print(3)
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Chroma_Mean_Price[key] * 0.0063939
        Used_keys_Chroma.append(df_Erwartungswert['Skin_Name'][pos])
        Chroma_Erw[df_Erwartungswert['Skin_Name'][pos]]=test
        
    elif df_Erwartungswert['Skin_Name'][pos] in Name_Chroma and df_Erwartungswert['Seltenheit_Militärstandard'][pos] == 1 and df_Erwartungswert['Kollektion_Chroma“'][pos] == 1:
        print(4)
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Chroma_Mean_Price[key] * 0.7992327
        Used_keys_Chroma.append(df_Erwartungswert['Skin_Name'][pos])
        Chroma_Erw[df_Erwartungswert['Skin_Name'][pos]]=test
    
    elif df_Erwartungswert['Skin_Name'][pos] in Name_Chroma and df_Erwartungswert['Seltenheit_Limitiert'][pos] == 1 and df_Erwartungswert['Kollektion_Chroma“'][pos] == 1:
        print(5)
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Chroma_Mean_Price[key] * 0.1598465
        Used_keys_Chroma.append(df_Erwartungswert['Skin_Name'][pos])
        Chroma_Erw[df_Erwartungswert['Skin_Name'][pos]]=test
    
    else:
        print(9)
        continue
    print(key)
    print(test)
    
    
Knife_avg_Chroma = (2.0424578625 + 1.1878436625 + 1.2978417375 + 0.6967269375 + 0.7645774124999999 + 0.5763837749999999 + 1.0362734249999999 + 0.8480158499999999 + 0.839755125 + 
      0.5449393124999999 + 0.4155042375 + 0.6928906875 + 0.392141475 + 0.33480232499999996 + 0.3550705125 + 0.42849633749999994 + 0.24457372499999996 + 0.2330777625 + 
      3.30270435 + 1.9431117749999998 + 2.262876 + 1.5221216999999998 + 1.4613682874999998 + 1.3683264375 + 3.1272342749999997 + 1.9599784875 + 2.0417417625 + 0.88699215+ 
      0.9577326 + 0.8431821749999999) / 30
Covert_avg_Chroma = (0.5195619201 + 0.23323668420000002)/2
Classified_avg_Chroma = (0.289322165 + 0.1868605585 + 0.576208048862069)/3
Restricted_avg_Chroma = (1.1118465835714286 + 0.43558171 + 0.5607246960526316 + 0.16304343000000002)/4
MilSpec_avg_Chroma = (1.2192250924615384 + 0.7249625393414635 + 0.45556263900000005 + 0.1718350305 + 0.15984654)/5
    
Expec_Val_Chroma = Knife_avg_Chroma + Covert_avg_Chroma + Classified_avg_Chroma + Restricted_avg_Chroma + MilSpec_avg_Chroma
print(Expec_Val_Chroma)  

#2.9949064381873054
 



#-----------Find all BreakoutItems-----------

Breakout_Positionen=[]
Breakout = np.where(df['Kollektion_Breakout“'] == 1)
Breakout_Positionen = Breakout[0].tolist()

#-----------Gettin SkinNames Breakout-----------

Skin_Name_Break=[]

for arr in Breakout_Positionen:
    if df_Erwartungswert['Skin_Name'][arr] in Skin_Name_Break:
        continue
    else:
        Skin_Name_Break.append(df['Skin_Name'][arr])
                
        
#-----------Calculating MeanValue Breakout-----------

Break_Mean_Price = dict.fromkeys(Skin_Name_Break, None)

for Name in Skin_Name_Break:
    Name_Index = df_Erwartungswert.loc[df_Erwartungswert.Skin_Name == Name].index.tolist()
    price = []
    for num in Breakout_Positionen:
        if num in Name_Index:
            price.append(df_Erwartungswert['Preis'][num])
    print(price)
    
    average = statistics.mean(price)
    print(average)
    Break_Mean_Price[Name] = average
    
        
#-----------Missing Skins-----------
#siehe: https://csgostash.com/case/18/Operation-Breakout-Weapon-Case?Knives=1

'''
Mil-Spec (Blue) – 79,92327% = 0.7992327
Restricted (Purple) – 15,98465% = 0.1598465
Classified (Pink) – 3,19693% = 0.0319693
Covert (Red) – 0,63939% = 0.0063939
Special Items/Knives (Gold) – 0,25575% = 0.0025575

'''
#Already expected value

Break_Mean_Price2 ={'Vanilla':3.793309575, 'Slaughter': 3.4344795374999997, 'Crimson Web':3.3347881874999996, 
                    'Stained': 1.77587685, 'Assimov':0.13945095899999999, 'Supernova':0.069053688, 'Osiris':0.10869562, 
                    'Abyss':1.27877232, 'Labyrinth':0.119884905, 'Ivory': 0.143861886}


print(0.18 * 0.7992327)

#-----------Calculating expected value Breakout-----------

Name=[]
for key in Break_Mean_Price:
    Name.append(key)

Used_keys=[]

Break_Erwart={}

for pos in Breakout_Positionen:
    if df_Erwartungswert['Skin_Name'][pos] in Used_keys:
        continue
    if df_Erwartungswert['Skin_Name'][pos] in Name and df_Erwartungswert['Seltenheit_Verdeckt'][pos] == 1 and df_Erwartungswert['Gruppe_Messer'][pos] == 1 and df_Erwartungswert['Kollektion_Breakout“'][pos] == 1:
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Break_Mean_Price[key] * 0.0025575
        Used_keys.append(df_Erwartungswert['Skin_Name'][pos])
        Break_Erwart[df_Erwartungswert['Skin_Name'][pos]]=test
        
    elif df_Erwartungswert['Skin_Name'][pos] in Name and df_Erwartungswert['Seltenheit_Vertraulich'][pos] == 1 and df_Erwartungswert['Kollektion_Breakout“'][pos] == 1:
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Break_Mean_Price[key] * 0.0319693
        Used_keys.append(df_Erwartungswert['Skin_Name'][pos])
        Break_Erwart[df_Erwartungswert['Skin_Name'][pos]]=test
    
    elif df_Erwartungswert['Skin_Name'][pos] in Name and df_Erwartungswert['Seltenheit_Verdeckt'][pos] == 1 and df_Erwartungswert['Kollektion_Breakout“'][pos] == 1:
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Break_Mean_Price[key] * 0.0063939
        Used_keys.append(df_Erwartungswert['Skin_Name'][pos])
        Break_Erwart[df_Erwartungswert['Skin_Name'][pos]]=test
        
    elif df_Erwartungswert['Skin_Name'][pos] in Name and df_Erwartungswert['Seltenheit_Militärstandard'][pos] == 1 and df_Erwartungswert['Kollektion_Breakout“'][pos] == 1:
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Break_Mean_Price[key] * 0.7992327
        Used_keys.append(df_Erwartungswert['Skin_Name'][pos])
        Break_Erwart[df_Erwartungswert['Skin_Name'][pos]]=test
    
    elif df_Erwartungswert['Skin_Name'][pos] in Name and df_Erwartungswert['Seltenheit_Limitiert'][pos] == 1 and df_Erwartungswert['Kollektion_Breakout“'][pos] == 1:
        key = df_Erwartungswert['Skin_Name'][pos]
        test = Break_Mean_Price[key] * 0.1598465
        Used_keys.append(df_Erwartungswert['Skin_Name'][pos])
        Break_Erwart[df_Erwartungswert['Skin_Name'][pos]]=test
    
    else:
        continue
    print(key)
    print(test)


Knife_avg_Breakout = (3.793309575 + 3.4344795374999997 + 2.7407229535714284 + 1.6612099166666667 + 3.3347881874999996 + 1.93954406 + 1.77587685 + 2.8824047999999998 + 1.8853762125 + 
                      2.97179795 + 1.28644808 + 1.240106175 + 3.7814287087499996)/13
Covert_avg_Breakout = (0.29461123846153847 + 0.13945095899999999)/2
Classified_avg_Breakout = (0.2827893808167539 + 0.34883621387999997 + 0.29904616041666665)/3
Restricted_avg_Breakout = (0.9489553883333334 + 0.069053688 + 0.10869562 + 0.485551908125)/4
MilSpec_avg_Breakout = (1.27877232 + 0.143861886 + 0.119884905 + 0.48146290724598934 + 0.23069986587640448)/5


Expec_Val_Breakout = Knife_avg_Breakout + Classified_avg_Breakout + Restricted_avg_Breakout + MilSpec_avg_Breakout
print(Expec_Val_Breakout)

#3.681723908347748





