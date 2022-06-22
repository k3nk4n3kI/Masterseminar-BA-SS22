#-----------Imports-----------
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import os
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn.model_selection import RandomizedSearchCV # cross validation
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plot
import shap


#-----------XGBoost Complete-----------

#Loading DF to PD

df_scraper = pd.read_csv('/Users/lars/Documents/Uni/Seminar_BA/Daten_Scraper_Clean.csv')

plot.boxplot(df_scraper['Preis'])
plot.show()


#-----------Train/Test Split-----------



X, y = df_scraper.drop(['Preis', 'Unnamed: 0.1', 'Unnamed: 0'], axis=1), df_scraper['Preis']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = 20)
Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size = 0.2, random_state = 20)

eval_set = [(Xtrain, ytrain),(Xval, yval)]

#-----------Hyperparameter Tuning-----------

model=XGBRegressor(n_estimaor=75)

param={
    'max_depth' : [4, 5, 6, 7, 8, 9, 10],
    'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9],
    'gamma' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    'reg_alpha' : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],                                  #Typicly between 0-1
    'reg_lambda' : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],                                 #Typicly between 0-1
    'learning_rate' : [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
}

optimal_param=RandomizedSearchCV(
    model,
    param_distributions=param,
    verbose=2,
    scoring='neg_root_mean_squared_error',
    cv=3,
    n_iter=13,

)

model_tuned=optimal_param.fit(
          Xtrain,
          ytrain,
          eval_metric='rmse',
          eval_set=eval_set,
          #estimator=70,
          early_stopping_rounds=10,
          verbose=True)

print(model_tuned.best_params_)


#{'subsample': 0.8, 'reg_lambda': 0.5, 'reg_alpha': 0.2, 'max_depth': 4, 'learning_rate': 0.05, 'gamma': 0.05}

#-----------Final Modell-----------


xgb_reg = XGBRegressor(
    max_depth=4,
    learning_rate=0.05,
    n_estimator=100,
    verbosity=2,
    gamma=0.05,
    subsample=0.8,
    reg_lambda=0.5,
    reg_alpha=0.2,
)

model = xgb_reg.fit(Xtrain,
          ytrain,
          eval_metric=['rmse'],
          eval_set=eval_set,
          early_stopping_rounds=10,
          verbose=True)

#TrainRMSE: 10.74350 ValRMSE: 12.00624


#-----------Predictions-----------


pre=model.predict(Xtest)
predictions = pre.tolist()

#-----------Plots-----------

results=model.evals_result()
print(results)

epochs = len(results['validation_0']['rmse'])
x_axis=range(0, epochs)
fig, ax = plot.subplots(figsize=(16,10))
ax.plot(x_axis, results['validation_0']['rmse'], c='b', label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], c='r', label='Val')
ax.legend()
plot.xlabel('Epochs')
plot.ylabel('RMSE')
plot.title('XGBoost RMSE')
plot.show()

#-----------Shap Values-----------

explainer = shap.Explainer(model, Xtest)
shap_values = explainer(Xtest)

#Most important features by mean

shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values, max_display=10)
#shap.summary_plot(shap_values, max_display=10)
#shap.dependence_plot('Abnutzung', shap_values[1], Xtest)
shap.plots.scatter(shap_values[:, "Preis_Sug"], color=shap_values)
shap.plots.scatter(shap_values[:, "Rabatt"])
shap.plots.scatter(shap_values[:, 'Kollektion_Vanguard“'])
shap.plots.scatter(shap_values[:, "MSV"])
shap.plots.waterfall(shap_values[600])


#-----------XGBoost PCA-----------

XGB_PCA_Feat = ['Titel1', 'Preis_Sug', 'Gruppe_Pass', 'Kollektion_Arms Deal 2“', 'Modell_Atlanta 2017 – Legenden (Holo/Glanz)', 'Modell_Wiederkehrende Herausforderer | Boston 2018', 'Modell_Legenden | London 2018', 'Kollektion_Sonnenaufgang“', 'Modell_Kistenschlüssel „Gamma 2“', 'Modell_Zuschauerpass – Berlin 2019', 'Modell_Community-Aufkleberkapsel 1', 'Modell_Kattowitz 2019 – Minor-Herausforderer (Holo/Glanz)', 'Gruppe_Geschenk', 'Modell_Kistenschlüssel „Gefahrenzone“', 'Modell_G2 Esports | Atlanta 2017', 'Modell_Anstecknadel: Guardian 3', 'Modell_AWOLNATION I Am', 'Modell_Kistenschlüssel „CS20“', 'Modell_Kattowitz 2019 – Wiederkehrende Herausforderer (Holo/Glanz)', 'Modell_Anstecknadel: Death Sentence', 'Gruppe_Pistole',
                'Modell_Antwerp 2022 Legends Autograph Capsule', 'Modell_bbno$ u mad!', 'Modell_Kistenschlüssel „Gamma“', 'Modell_Krakau 2017 – Herausforderer (Holo/Glanz)', 'Modell_Kistenschlüssel „Jagd“', 'Modell_Anstecknadel: Baggage', 'Modell_Kistenschlüssel „Falchion“', 'Gruppe_Schwer', 'Modell_Anstecknadel: Wächter', 'Modell_Anstecknadel: Easy Peasy', 'Modell_Team SoloMid | Köln 2015', 'Modell_Antwerp 2022 Overpass Souvenir Package', 'Modell_Kistenschlüssel „Horizont“', 'Modell_Kattowitz 2019 – Legenden (Holo/Glanz)', 'Modell_London 2018 – Wiederkehrende Herausforderer (Holo/Glanz)', 'Modell_Kistenschlüssel „Operation Vanguard“', 'Modell_Premiumpass für Operation Springflut', 'Status1_Verkratzt', 'Modell_London 2018 – Minor-Herausforderer (Holo/Glanz)',
                'Modell_Gendarmerie Nationale', 'Kollektion_Assault“', 'Modell_Anstecknadel: Cobblestone', 'Modell_Flipsid3 Tactics | Köln 2015', 'Modell_Kapsel: Bestiary', 'Modell_Anstecknadel: Canals', 'Modell_Anstecknadel: Hydra', 'Seltenheit_Bemerkenswert', 'Modell_Team Liquid | MLG Columbus 2016', 'Modell_Kistenschlüssel „E-Sport“', 'Modell_Anstecknadel: Italy', 'Modell_Team Immunity | Köln 2015', 'Modell_FBI-SWAT', 'Modell_Daniel Sadowski Eye of the Dragon', 'Modell_Antwerp 2022 Contenders Sticker Capsule', 'Modell_Anstecknadel: Overpass', 'Modell_Anstecknadel: Chroma', 'Modell_Anstecknadel: Nuke', 'Modell_Schlüssel: Community-Aufkleberkapsel 1', 'Modell_Anstecknadel: Lambda', 'Modell_Teilnehmende Legenden | Boston 2018', 'Modell_Anstecknadel: Kopfkrabbenglyphe',
                'Modell_Anstecknadel: Combine-Helm', 'Modell_Renegades | Köln 2015', 'Kollektion_Zerbrochener Reißzahn“-Agenten', 'Modell_Anstecknadel: Bloodhound', 'Modell_Troels Folmann Uber Blasto Phone', 'Modell_Anstecknadel: Guardian 2', 'Modell_Anstecknadel: Militia', 'Modell_Gruppe D (Glanz) | Köln 2015', 'Modell_Legenden (Glanz) | Klausenburg 2015', 'Modell_DreamHack Klausenburg 2015 – Legenden (Glanz)', 'Modell_Hühnerkapsel', 'Kollektion_Baggage“', 'Modell_Aufnäherpaket zu Half-Life: Alyx', 'Kollektion_E-Sport Sommer 2014“', 'Modell_RMR 2020 – Herausforderer', 'Modell_Anstecknadel: Cache', 'Modell_Anstecknadel: CMB', 'Modell_Team EnVyUs | Köln 2015', 'Modell_Legenden (Glanz) | Atlanta 2017', 'Modell_Anstecknadelkapsel der zweiten Serie', 'Modell_Antwerp 2022 Legends Sticker Capsule',
                'Modell_Anstecknadelkapsel zu Half-Life: Alyx', 'Modell_Vexed Gaming | Klausenburg 2015', 'Modell_Souvenirpaket: Berlin 2019 – Inferno', 'Modell_Waffenkiste „E-Sport Sommer 2014“', 'Modell_Anstecknadel: Mirage', 'Modell_Anstecknadel: Kupfer-Lambda', 'Modell_London 2018 – Legenden (Holo/Glanz)', 'Modell_Antwerp 2022 Challengers Sticker Capsule', 'Modell_Michael Bross Invasion!', 'Modell_Team Liquid | Klausenburg 2015', 'Modell_Boston 2018 – Minor-Herausforderer mit Flash Gaming (Holo/Glanz)', 'Modell_Team eBettle | Köln 2015', 'Modell_Antwerp 2022 Nuke Souvenir Package', 'Seltenheit_Standardqualität', 'Modell_Waffenkiste „E-Sport Winter 2013“', 'Modell_Kistenschlüssel „Clutch“', 'Modell_Fnatic | Klausenburg 2015', 'Modell_Souvenirpaket: IEM Kattowitz 2019 – Overpass', 'Modell_Mord Fustang Diamonds',
                'Modell_Gruppe C (Glanz) | Köln 2015', 'Modell_Krakau 2017 – Legenden (Holo/Glanz)', 'Modell_Antwerp 2022 Inferno Souvenir Package', 'Modell_Souvenirpaket: FACEIT London 2018 – Inferno', 'Kollektion_Jagd“', 'Modell_NZSAS', 'Modell_Team EnVyUs | MLG Columbus 2016', 'Modell_"Ian Hultquist Lions Mouth"', 'Modell_GODSENT | Atlanta 2017', 'Modell_Team EnVyUs | Atlanta 2017', 'Modell_Virtus.Pro | MLG Columbus 2016', 'Modell_Flipsid3 Tactics | Köln 2016', 'Modell_Antwerp 2022 Vertigo Souvenir Package', 'Modell_Austin Wintory Desert Fire', 'Modell_Anstecknadel: Bravo', 'Modell_Anstecknadel: Office', 'Modell_Souvenirpaket: Atlanta 2017 – Cache', 'Modell_mousesports | Köln 2015', 'Modell_Kistenschlüssel „Operation Phoenix“', 'Modell_Gruppe B (Glanz) | Köln 2015', 'Modell_Flipsid3 Tactics | MLG Columbus 2016',
                'Modell_Communitykapsel 2018', 'Kollektion_Chroma“', 'Modell_Titan | Köln 2015', 'Modell_Antwerp 2022 Viewer Pass', 'Modell_Souvenirpaket: FACEIT London 2018 – Dust II', 'Modell_Boston 2018 – Minor-Herausforderer (Holo/Glanz)', 'Modell_Aufkleberkapsel „Battlefield 2042“', 'Modell_Souvenirpaket: FACEIT London 2018 – Mirage', 'Modell_Ninjas in Pyjamas | MLG Columbus 2016', 'Modell_Waffenkiste „Chroma“', 'Modell_Natus Vincere | Atlanta 2017', 'Modell_Splyce | MLG Columbus 2016', 'Modell_Flipsid3 Tactics | Klausenburg 2015', 'Modell_Waffenkiste „E-Sport 2013“', 'Modell_Various Artists Hotline Miami', 'Modell_Kelly Bailey Hazardous Environments', 'Modell_Souvenirpaket: FACEIT London 2018 – Train', 'Seltenheit_Überlegen', 'Modell_Astralis | MLG Columbus 2016', 'Modell_Souvenirpaket: IEM Kattowitz 2019 – Dust II',
                'Modell_Zugangspass für Operation Hydra', 'Modell_Souvenirpaket: Atlanta 2017 – Mirage', 'Modell_Natus Vincere | MLG Columbus 2016', 'Modell_Kistenschlüssel „Handschuhe“', 'Modell_Herausforderer | Krakau 2017', 'Modell_G2 Esports | MLG Columbus 2016', 'Modell_Souvenirpaket: IEM Kattowitz 2019 – Train', 'Modell_ESL One Köln 2015 – Legenden (Glanz)', 'Modell_Souvenirpaket: ELEAGUE Boston 2018 – Mirage', 'Modell_mousesports | MLG Columbus 2016', 'Modell_Souvenirpaket: ESL One Köln 2015 – Cobblestone', 'Modell_Fnatic | MLG Columbus 2016', 'Modell_Luminosity Gaming | MLG Columbus 2016', 'Modell_Mateo Messina For No Mankind', 'Modell_Beartooth Disgusting', 'Modell_Kelly Bailey', 'Modell_Daniel Sadowski Total Domination', 'Modell_MP7', 'Modell_Sasha LNOE', 'Modell_Darude Moments CSGO', 'Modell_Skog II-Headshot',
                'Modell_Virtus.Pro | Köln 2015', 'Modell_Sean Murray A*D*8', 'Modell_bbno$ u mad! bbno$ u mad!', 'Modell_Robert Allaire Insurgency', 'Kollektion_Arms Deal“', 'Modell_Team EnVyUs | Klausenburg 2015', 'Modell_Flashbang Dance', 'Modell_Hazardous Environments', 'Modell_Damjan Mravunac The Talos Principle', 'Modell_HellRaisers | Atlanta 2017', 'Modell_Community-Graffitibox 1', 'Modell_Souvenirpaket: IEM Kattowitz 2019 – Nuke', 'Kollektion_Falchion“', 'Modell_SEAL-Kampfschwimmer', 'Kollektion_Vertigo“', 'Kollektion_Vertigo\\xa02021“', 'Kollektion_Bravo“', 'Kollektion_Götter und Monster“', 'Modell_Souvenirpaket: IEM Kattowitz 2019 – Inferno', 'Kollektion_Nuke“', 'Kollektion_Durchbruch"', 'Zustand_Agent', 'Kollektion_Vanguard“', 'Kollektion_Phoenix“', 'Kollektion_Cache“', 'Kollektion_Nordisch“', 'Modell_Waffenkiste „Operation Springflut“',
                'Kollektion_CS20“', 'Kollektion_Zerfetztes Netz“', 'Kollektion_Schlangenbiss”', 'Namensschild', 'Modell_M4A1-S', 'Kollektion_Canals“', 'Modell_Waffenkiste „Chroma 3“', 'Extras_StatTrak™', 'Modell_Souvenirpaket: ELEAGUE Boston 2018 – Inferno', 'Kollektion_Inferno“', 'Kollektion_Overpass“', 'Sterne', 'Kollektion_Horizont“', 'Modell_Waffenkiste „Operation Vanguard“', 'Seltenheit_Meister', 'Kollektion_Durchbruch“', 'Kollektion_Handschuhe“', 'Modell_Dual Berettas', 'Zustand_Sammlerstück', 'Modell_Waffenkiste „Horizont“', 'Kollektion_Spektrum 2“', 'Kollektion_Hydra“', 'Modell_Waffenkiste „Operation Hydra“', 'Modell_M249', 'Modell_Waffenkiste „Handschuhe“', 'Kollektion_Gamma 2“', 'Modell_Waffenkiste „Gamma 2“', 'Kollektion_Operation Springflut“', 'Seltenheit_Verbraucherklasse', 'Modell_Waffenkiste „Gamma“', 'Zustand_Einsatzerprobt', 'Modell_CZ75-Auto',
                'Zustand_Minimale', 'Kollektion_Mirage“', 'Modell_MP5-SD', 'Zustand_Musikkit', 'Modell_Waffenkiste „Gefahrenzone“', 'Seltenheit_Vertraulich', 'Modell_Kapsel „Halo“', 'Seltenheit_Limitiert', 'Modell_RMR 2020 – Legenden', 'Modell_FBI-Scharfschütze', 'Modell_P2000', 'Modell_Souvenirpaket: Berlin 2019 – Overpass', 'Kollektion_Chroma 2“', 'Gruppe_Schlüssel', 'Modell_Berlin 2019 – Minor-Herausforderer (Holo/Glanz)', 'Kollektion_Revolver“', 'Modell_Herausforderer (Glanz) | Atlanta 2017', 'Kollektion_Schatten“', 'Modell_Legenden | Berlin 2019', 'Modell_Minor-Herausforderer | Berlin 2019', 'Modell_Minor-Herausforderer | Kattowitz 2019', 'Modell_Wiederkehrende Herausforderer | Berlin 2019', 'Modell_Wiederkehrende Herausforderer | Kattowitz 2019', 'Gruppe_Beschriftungsschild', 'Kollektion_Winteroffensive“', 'Modell_Berlin 2019 – Legenden (Holo/Glanz)', 'Modell_Legenden | Kattowitz 2019',
                'Modell_Kistenschlüssel „Operation Hydra“', 'Modell_Anstecknadelkapsel der dritten Serie', 'Modell_Aufkleberkapsel zu Half-Life: Alyx']


#-----------Train/Test Split-----------

X_pca, y_pca = df_scraper[XGB_PCA_Feat], df_scraper['Preis']
Xtrain_pca, Xtest_pca, ytrain_pca, ytest_pca = train_test_split(X_pca, y_pca, test_size=0.2, random_state = 20)
Xtrain_pca, Xval_pca, ytrain_pca, yval_pca = train_test_split(Xtrain_pca, ytrain_pca, test_size = 0.2, random_state = 20)

eval_set_pca = [(Xtrain_pca, ytrain_pca),(Xval_pca, yval_pca)]


#-----------Hyperparameter Tuning-----------

model_pca=XGBRegressor(n_estimaor=75)

param={
    'max_depth' : [4, 5, 6, 7, 8, 9, 10],
    'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9],
    'gamma' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    'reg_alpha' : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],                                  #Typicly between 0-1
    'reg_lambda' : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],                                 #Typicly between 0-1
    'learning_rate' : [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
}

optimal_param_pca=RandomizedSearchCV(
    model,
    param_distributions=param,
    verbose=2,
    scoring='neg_root_mean_squared_error',
    cv=3,
    n_iter=13,

)

model_tuned_pca=optimal_param.fit(
          Xtrain_pca,
          ytrain_pca,
          eval_metric='rmse',
          eval_set=eval_set_pca,
          #estimator=70,
          early_stopping_rounds=10,
          verbose=True)

print(model_tuned_pca.best_params_)

#{'subsample': 0.9, 'reg_lambda': 0.7, 'reg_alpha': 0.6, 'max_depth': 4, 'learning_rate': 0.07, 'gamma': 0.06}

#-----------Final Modell-----------


xgb_reg_pca = XGBRegressor(
    max_depth=4,
    learning_rate=0.07,
    n_estimator=100,
    verbosity=2,
    gamma=0.06,
    subsample=0.9,
    reg_lambda=0.6,
    reg_alpha=0.6,
)

model_pca = xgb_reg.fit(Xtrain_pca,
          ytrain_pca,
          eval_metric=['rmse'],
          eval_set=eval_set_pca,
          early_stopping_rounds=10,
          verbose=True)

#TrainRMSE: 14.36379 ValRMSE: 16.10129



#-----------Predictions-----------


pre_pca=model_pca.predict(Xtest_pca)
predictions_pca = pre_pca.tolist()


#-----------Plots-----------

results_pca=model_pca.evals_result()
print(results)

epochs = len(results_pca['validation_0']['rmse'])
x_axis=range(0, epochs)
fig, ax = plot.subplots(figsize=(16,10))
ax.plot(x_axis, results_pca['validation_0']['rmse'], c='b', label='Train')
ax.plot(x_axis, results_pca['validation_1']['rmse'], c='r', label='Val')
ax.legend()
plot.xlabel('Epochs')
plot.ylabel('RMSE')
plot.title('XGBoost_PCA RMSE')
plot.show()

#-----------Shap Values-----------

explainer_pca = shap.Explainer(model, Xtest_pca)
shap_values_pca = explainer_pca(Xtest_pca)

shap.plots.bar(shap_values_pca)
shap.plots.beeswarm(shap_values_pca, max_display=10)
#shap.summary_plot(shap_values, max_display=10)
#shap.dependence_plot('Abnutzung', shap_values[1], Xtest)
shap.plots.scatter(shap_values_pca[:, "Preis_Sug"], color=shap_values_pca)
shap.plots.scatter(shap_values_pca[:, "Rabatt"])
shap.plots.scatter(shap_values_pca[:, "Sterne"])
shap.plots.scatter(shap_values_pca[:, "Namensschild"])



inds = shap.utils.potential_interactions(shap_values[:, "Seltenheit_Vertraulich"], shap_values)

#ake plots colored by each of the top three possible interacting features
for i in range(8):
    shap.plots.scatter(shap_values[:,"Seltenheit_Vertraulich"], color=shap_values[:,inds[i]])





















