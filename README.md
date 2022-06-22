# Masterseminar-BA-SS22

This is the repository for the Master-seminar Business Analytics. The topic of the seminar was Analyzing Counter-Strike: Global Offensive Skin Prices.

Goal of the paper was to find out which features are influencing the price of CS:GO skins the most. Therefor a scraper has to be programmed first. 
This can be found in the Scraper.py file. Afterwards the scraped data was perprocessed. Further more a PCA was applied on the preprocessed Data to 
reduce feature dimensonality. The preprocessing and the PCA can be found in Preprocessing.py. In the End two XGBoost Modells were build and trained. 
One on the complette preprocessed dataset and one on the dimension reduced Dataset. To find the most import features the avarage SHAP Values were
used to determine them. Both modells are located as well as the SHAP value calculation can be found in the XGBoost.py file.


Beside finding the most important features it was also investigated if it's economicly reasonable to buy and open weaponboxes in CS:GO. 
Therefor the price to open a case was compared to the expected value of the specific case. Only the most expensive and the cheapest weaponcase 
were investigated. The code can be found in the Markt_Efficency.py file.
