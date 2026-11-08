# Application Streamlit — Machine Learning, Prédictions

L'application lit les fichiers `.joblib` sauvegardés par les notebooks. Chaque modèle a 2 interfaces :
**prédiction simple** (formulaire) et **prédiction multiple** (fichier csv, tsv, txt, xlsx ou json).

## Arborescence
```
app.py
requirements.txt
models/
  best_model.joblib          encoders.joblib          scaler.joblib          uniques.joblib            <- Car_Data (Prix)
  best_model_dataset.joblib  encoders_dataset.joblib  scaler_dataset.joblib  uniques_dataset.joblib    <- Car_dataset (Selling_Price)
  bounds_dataset.joblib                                                                                  <- bornes d'outliers de Car_dataset
  best_model_tel.joblib      encoders_tel.joblib      scaler_tel.joblib      uniques_tel.joblib        <- Téléphones (etat)
```
Un jeu de données dont il manque des fichiers est signalé dans la barre latérale ; les autres restent utilisables.

## Modèles supplémentaires (optionnel)
Pour ajouter d'autres modèles que le meilleur, sauvegardez-les dans `models/` avec le même suffixe :
`lr_model`, `rr_model`, `lg_model`, `dt_model`, `rf_model` (ex. `rf_model_tel.joblib`). Ils apparaissent automatiquement chacun dans un onglet, et le meilleur est marqué ⭐ (`best_model` sert alors uniquement à repérer le meilleur).

## Configuration
Le début de `app.py` (`DATASETS`) décrit, pour chaque jeu de données, l'ordre des colonnes de `df` et les colonnes encodées
(ordre de `encoders.joblib`). Si vos notebooks ont un autre ordre de colonnes, modifiez-le là ; en cas de désaccord avec les
fichiers, l'application affiche un message précis au lieu de donner de mauvaises prédictions.

## Lancer / déployer
`pip install -r requirements.txt` puis `streamlit run app.py`.
Streamlit Cloud : poussez `app.py`, `requirements.txt` et le dossier `models/` sur GitHub.
Dans `requirements.txt`, mettez la **même version de scikit-learn** que dans Colab (`scikit-learn==X.Y.Z`).
