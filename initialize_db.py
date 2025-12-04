import pandas as pd
from sklearn import preprocessing
from sqlalchemy import create_engine

label_encoder = preprocessing.LabelEncoder()

raw_dataset = pd.read_csv("data/dataset.csv")

# Encodage

def etudes_to_number(x: str):
    if x == "aucun":
        return 0
    elif x == "bac":
        return 1
    elif x == "bac+2":
        return 2
    elif x == "master":
        return 3
    elif x == "doctorat":
        return 4
    else:
        return None


encoded_dataset = raw_dataset.copy()

encoded_dataset["prenom"] = label_encoder.fit_transform(encoded_dataset['prenom'])
encoded_dataset["nom"] = label_encoder.fit_transform(encoded_dataset['nom'])

encoded_dataset["niveau_etude"] = encoded_dataset.apply(lambda x: etudes_to_number(x["niveau_etude"]), axis=1)
encoded_dataset["date_creation_compte"] =  pd.to_datetime(encoded_dataset['date_creation_compte'])

## Autres catégories
encoded_dataset = pd.get_dummies(encoded_dataset)

# Nettoyage
cleaned_dataset = encoded_dataset.drop_duplicates(inplace=False)
cleaned_dataset = cleaned_dataset.fillna(value=cleaned_dataset.median(), inplace=False) 
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

final_dataset = final_dataset.drop(columns=["nom", "prenom", "sexe_H", "sexe_F", "nationalité_francaise_oui", "nationalité_francaise_non", "taille", "poids"
])
final_dataset = final_dataset[final_dataset.columns.drop(list(final_dataset.filter(regex='region_')))]


# Création de la BDD

engine = create_engine("sqlite:///database.db")

final_dataset.to_sql("loans", engine, if_exists="replace", index=True, index_label="id")

print("Le DataFrame a été inséré dans la table 'loans' de la base de données.")