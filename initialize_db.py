import pandas as pd
from sklearn import preprocessing
from sqlalchemy import create_engine

label_encoder = preprocessing.LabelEncoder()

raw_dataset = pd.read_csv("data/dataset.csv")

encoded_dataset = raw_dataset.copy()

encoded_dataset["date_creation_compte"] =  pd.to_datetime(encoded_dataset['date_creation_compte'])
encoded_dataset["date_creation_compte"] =  encoded_dataset["date_creation_compte"].apply(lambda x: x.timestamp())

## Autres catégories
# encoded_dataset = pd.get_dummies(encoded_dataset)

# Nettoyage
cleaned_dataset = encoded_dataset.drop_duplicates(inplace=False)

num_columns=["age", "historique_credits", "revenu_estime_mois", "risque_personnel", "score_credit", "loyer_mensuel", "date_creation_compte", "montant_pret"]

cleaned_dataset[num_columns] = cleaned_dataset[num_columns].fillna(value=cleaned_dataset[num_columns].median(), inplace=False) 


cleaned_dataset = cleaned_dataset.drop(cleaned_dataset.loc[cleaned_dataset["loyer_mensuel"] < 0.0].index, inplace=False)

threshold = 1.5

def find_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    outliers = cleaned_dataset[(col < Q1 - threshold * IQR) | (col > Q3 + threshold * IQR)]
    return outliers

poids_outliers = find_outliers(cleaned_dataset["poids"])
cleaned_dataset = cleaned_dataset.drop(poids_outliers.index)

taille_outliers = find_outliers(cleaned_dataset["taille"])
cleaned_dataset = cleaned_dataset.drop(taille_outliers.index)

revenu_outliers = find_outliers(cleaned_dataset["revenu_estime_mois"])
cleaned_dataset = cleaned_dataset.drop(revenu_outliers.index)

pret_outliers = find_outliers(cleaned_dataset["montant_pret"])
cleaned_dataset = cleaned_dataset.drop(pret_outliers.index)

# Mise en conformité
final_dataset = cleaned_dataset.copy()

final_dataset["imc"] = final_dataset.apply(lambda x: x["poids"] / ((x["taille"] / 100) ** 2), axis=1)
print(final_dataset)

final_dataset = final_dataset.drop(columns=["nom", "prenom", "sexe", "nationalité_francaise", "taille", "poids", "region"])


# Création de la BDD

engine = create_engine("sqlite:///database.db")


final_dataset.to_sql("loans", engine, if_exists="replace", index=True, index_label="id")

final_dataset.to_csv('data/cleaned_dataset.csv')

print("Le DataFrame a été inséré dans la table 'loans' de la base de données.")