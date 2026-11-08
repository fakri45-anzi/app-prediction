# Application Streamlit : prédictions avec les fichiers .joblib sauvegardés par les notebooks.
# Chaque modèle a 2 interfaces : prédiction simple (formulaire) et prédiction multiple (fichier).
#
# Fichiers lus dans le dossier models/ (ou à côté de app.py) :
#   best_model<suffixe>.joblib, encoders<suffixe>.joblib, scaler<suffixe>.joblib, uniques<suffixe>.joblib
#   (+ bounds<suffixe>.joblib si le notebook écrête les outliers)
#   (+ optionnel : lr_model, rr_model, lg_model, dt_model, rf_model <suffixe>.joblib pour ajouter d'autres modèles)
# Suffixes : ""  -> Car_Data (Prix) | "_dataset" -> Car_dataset (Selling_Price) | "_tel" -> Téléphones (etat)
import io
import math
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Machine Learning - Prédictions", page_icon="🤖", layout="wide")

FOLDER = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------
# CONFIGURATION : doit correspondre à vos notebooks (ordre des colonnes = ordre de df)
# kind : "cat" (catégorielle), "int" (entier) ou "float" (décimal)
# encoder_cols : colonnes encodées, dans l'ordre de cat_data.columns (= ordre de encoders.joblib)
# ----------------------------------------------------------------------------------
DATASETS = {
    "car_data": dict(
        title="🚗 Voitures (Dakar) — Prix", suffix="", task="regression",
        target="Prix", unit="FCFA", log_target=True,       # le modèle prédit log(1 + Prix)
        features=[("Marque", "cat"), ("Année", "int"), ("Boite_Vitesse", "cat"),
                  ("Adresse", "cat"), ("Venant_Occasion", "cat")],
        encoder_cols=["Marque", "Boite_Vitesse", "Adresse", "Venant_Occasion"],
        expect={"Année": (1950, 2035)},
    ),
    "car_dataset": dict(
        title="🚙 Voitures — Selling Price", suffix="_dataset", task="regression",
        target="Selling_Price", unit="", log_target=False,
        features=[("Kms_Driven", "int"), ("Present_Price", "float"), ("Fuel_Type", "cat"),
                  ("Seller_Type", "cat"), ("Transmission", "cat"), ("Age", "int")],
        encoder_cols=["Fuel_Type", "Seller_Type", "Transmission"],
        expect={"Age": (0, 40)},
    ),
    "telephone": dict(
        title="📱 Téléphones — état de l'appareil", suffix="_tel", task="classification",
        target="etat", unit="", log_target=False,
        features=[("prix", "int"), ("adresse", "cat"), ("marque", "cat"),
                  ("dim_ecr", "float"), ("ram", "int"), ("stockage", "int")],
        encoder_cols=["adresse", "marque", "etat"],         # etat = cible (dernier encodeur)
        expect={},
    ),
}
EXTRA_MODELS = {"lr_model": "Régression linéaire", "rr_model": "Ridge", "lg_model": "Régression logistique",
                "dt_model": "Arbre de décision", "rf_model": "Random Forest"}
FILE_TYPES = ["csv", "tsv", "txt", "xlsx", "json"]


# ----------------------------------------------------------------------------------
# Chargement des objets sauvegardés (une seule fois grâce au cache)
# ----------------------------------------------------------------------------------
def find_file(stem, suffix):
    for folder in (FOLDER / "models", FOLDER):
        p = folder / f"{stem}{suffix}.joblib"
        if p.exists():
            return p
    return None


def missing_files(cfg):
    return [f"{s}{cfg['suffix']}.joblib" for s in ("best_model", "encoders", "scaler", "uniques")
            if find_file(s, cfg["suffix"]) is None]


@st.cache_resource
def load_objects(ds):
    """Charge les fichiers d'un jeu de données et vérifie qu'ils sont cohérents avec la configuration.
    Retourne (objets, liste_de_problèmes)."""
    cfg, sfx, problems = DATASETS[ds], DATASETS[ds]["suffix"], []
    load = lambda stem: joblib.load(find_file(stem, sfx))
    encoders_list, uniques, scaler, best = load("encoders"), load("uniques"), load("scaler"), load("best_model")
    bounds_raw = joblib.load(find_file("bounds", sfx)) if find_file("bounds", sfx) else {}

    names = [n for n, _ in cfg["features"]]
    enc_cols = cfg["encoder_cols"]
    if len(encoders_list) != len(enc_cols):
        problems.append(f"encoders{sfx}.joblib contient {len(encoders_list)} encodeur(s) mais {len(enc_cols)} sont "
                        f"attendus ({', '.join(enc_cols)}). Premières valeurs de uniques{sfx}.joblib : "
                        + " | ".join(str(u[:3]) for u in uniques))
        return None, problems
    encoders = dict(zip(enc_cols, encoders_list))
    for c, u, e in zip(enc_cols, uniques, encoders_list):
        if len(u) != len(e.classes_):
            problems.append(f"{c} : uniques et encodeur ne correspondent pas.")
    if getattr(scaler, "n_features_in_", len(names)) != len(names):
        problems.append(f"scaler{sfx}.joblib attend {scaler.n_features_in_} variables, la configuration en a {len(names)}.")
    if getattr(best, "n_features_in_", len(names)) != len(names):
        problems.append(f"best_model{sfx}.joblib attend {best.n_features_in_} variables, la configuration en a {len(names)}.")
    if problems:
        return None, problems

    # statistiques des variables numériques déduites du scaler / des bornes
    feats = []
    for j, (n, kind) in enumerate(cfg["features"]):
        f = {"name": n, "kind": "cat" if kind == "cat" else "num", "is_int": kind == "int",
             "min": None, "max": None, "median": 0.0}
        if f["kind"] == "num":
            if hasattr(scaler, "center_"):                          # RobustScaler : médiane
                f["median"] = float(scaler.center_[j])
            elif hasattr(scaler, "data_min_"):                      # MinMaxScaler : min / max observés
                f["min"], f["max"] = float(scaler.data_min_[j]), float(scaler.data_max_[j])
                f["median"] = (f["min"] + f["max"]) / 2
            b = bounds_raw.get(j, bounds_raw.get(n)) if isinstance(bounds_raw, dict) else None
            if b is not None:
                f["min"], f["max"] = float(b[0]), float(b[1])
            rng = cfg["expect"].get(n)
            if rng and hasattr(scaler, "center_") and not (rng[0] <= f["median"] <= rng[1]):
                problems.append(f"Ordre des colonnes suspect : la médiane de « {n} » dans le scaler est "
                                f"{f['median']:g} (attendu entre {rng[0]} et {rng[1]}). Vérifiez `features` dans app.py.")
        feats.append(f)
    if problems:
        return None, problems

    bounds = {}
    if isinstance(bounds_raw, dict):
        for k, v in bounds_raw.items():
            n = names[k] if isinstance(k, (int, np.integer)) else k
            bounds[n] = (float(v[0]), float(v[1]))

    # modèles : tous ceux trouvés (lr_model, rr_model, lg_model, dt_model, rf_model) ; le meilleur est marqué ⭐.
    # S'il n'y en a aucun, on utilise best_model seul.
    models = {}
    for stem, label in EXTRA_MODELS.items():
        p = find_file(stem, sfx)
        if p is not None:
            m = joblib.load(p)
            is_best = type(m) is type(best) and m.get_params() == best.get_params()
            models[f"{label} ⭐" if is_best else label] = m
    if not models:
        models = {f"Meilleur modèle ({type(best).__name__})": best}
    obj = dict(cfg=cfg, features=feats, encoders={c: e for c, e in encoders.items() if c != cfg["target"]},
               bounds=bounds, scaler=scaler, models=models, best=next(iter(models)),
               class_names=list(encoders[cfg["target"]].classes_) if cfg["task"] == "classification" else None)
    return obj, []


# ----------------------------------------------------------------------------------
# Prétraitement et prédiction (mêmes étapes que dans les notebooks)
# ----------------------------------------------------------------------------------
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def prepare(df, o):
    """Tableau brut -> matrice encodée, écrêtée puis normalisée. Retourne (matrice, erreurs, lignes_valides)."""
    names = [f["name"] for f in o["features"]]
    cols = {norm(c): c for c in df.columns}
    missing = [n for n in names if norm(n) not in cols]
    if missing:
        raise ValueError("Colonnes manquantes dans le fichier : " + ", ".join(missing)
                         + " — colonnes attendues : " + ", ".join(names))
    X = np.full((len(df), len(names)), np.nan)
    errors = [[] for _ in range(len(df))]
    for j, f in enumerate(o["features"]):
        col = df[cols[norm(f["name"])]]
        if f["kind"] == "cat":
            lookup = {norm(c): i for i, c in enumerate(o["encoders"][f["name"]].classes_)}
            codes = col.map(lambda v: lookup.get(norm(v), np.nan)).astype(float)
            for i in np.where(codes.isna())[0]:
                errors[i].append(f"{f['name']} : valeur inconnue « {col.iloc[i]} »")
        else:
            if not pd.api.types.is_numeric_dtype(col):             # '6,5' -> 6.5 ; '580 000' -> 580000
                col = col.astype(str).str.replace(r"[\s\u00a0]", "", regex=True).str.replace(",", ".", regex=False)
            codes = pd.to_numeric(col, errors="coerce").astype(float)
            for i in np.where(codes.isna())[0]:
                errors[i].append(f"{f['name']} : valeur manquante ou non numérique")
        X[:, j] = codes.to_numpy()
    ok = np.array([len(e) == 0 for e in errors], dtype=bool)
    err = pd.Series(["; ".join(e) for e in errors], index=df.index)
    X = X[ok]
    sc = o["scaler"]
    for j, f in enumerate(o["features"]):
        if f["kind"] != "num":
            continue
        if f["name"] in o["bounds"]:                               # écrêtage des outliers (bornes du notebook)
            X[:, j] = np.clip(X[:, j], *o["bounds"][f["name"]])
        elif hasattr(sc, "data_min_"):                             # MinMaxScaler : rester dans la plage vue à l'entraînement
            X[:, j] = np.clip(X[:, j], sc.data_min_[j], sc.data_max_[j])
    return (sc.transform(X) if ok.any() else X), err, ok


def Pred_func(df, o, model_name, with_proba=True):
    """Prédit pour toutes les lignes de df avec le modèle choisi."""
    model, cfg = o["models"][model_name], o["cfg"]
    X, err, ok = prepare(df, o)
    n, proba = len(df), None
    if cfg["task"] == "classification":
        pred = np.full(n, None, dtype=object)
        if ok.any():
            pred[ok] = [o["class_names"][int(k)] for k in model.predict(X)]
            if with_proba and hasattr(model, "predict_proba"):
                proba = pd.DataFrame(np.nan, index=df.index, columns=o["class_names"])
                p = model.predict_proba(X)
                for k, c in enumerate(model.classes_):
                    proba.iloc[np.where(ok)[0], int(c)] = p[:, k]
    else:
        pred = np.full(n, np.nan)
        if ok.any():
            y = model.predict(X)
            if cfg["log_target"]:
                y = np.expm1(y)
            pred[ok] = np.clip(y, 0, None)
    return pred, proba, ok, err


def Pred_func_csv(df, o, model_name):
    """Prédiction multiple : fichier d'origine + colonne de prédiction (+ confiance, + erreurs)."""
    pred, proba, ok, err = Pred_func(df, o, model_name)
    out = df.copy()
    out[f"{o['cfg']['target']}_prediction"] = pred
    if proba is not None:
        out["confiance"] = proba.max(axis=1).round(4).to_numpy()
    if not ok.all():
        out["erreur"] = err.to_numpy()
    return out, ok


def read_file(file):
    raw, name = file.getvalue(), file.name.lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(raw))
    elif name.endswith(".json"):
        df = pd.read_json(io.BytesIO(raw))
    else:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]
    return df.reset_index(drop=True)


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.0f}".replace(",", " ") if abs(v) >= 1000 else f"{v:,.2f}"


def unit_of(o):
    return f" {o['cfg']['unit']}" if o["cfg"]["unit"] else ""


# ----------------------------------------------------------------------------------
# Interface 1 : prédiction simple
# ----------------------------------------------------------------------------------
def widget_for(f, o, key):
    if f["kind"] == "cat":
        return st.selectbox(f["name"], list(o["encoders"][f["name"]].classes_), key=key)
    help_txt = (f"Plage observée : {f['min']:g} – {f['max']:g}" if f["min"] is not None else None)
    if f["is_int"]:
        return st.number_input(f["name"], value=max(int(round(f["median"])), 0), step=1, min_value=0,
                               help=help_txt, key=key)
    return st.number_input(f["name"], value=max(float(f["median"]), 0.0), step=0.1, min_value=0.0,
                           help=help_txt, key=key)


def simple_interface(ds, o, model_name):
    cfg = o["cfg"]
    st.write(f"Renseignez les caractéristiques puis cliquez sur **Prédire** (cible : *{cfg['target']}*).")
    with st.form(f"form_{ds}_{model_name}"):
        cols = st.columns(2)
        values = {}
        for i, f in enumerate(o["features"]):
            with cols[i % 2]:
                values[f["name"]] = widget_for(f, o, f"{ds}_{model_name}_{f['name']}")
        submit = st.form_submit_button("Prédire")

    if submit:
        row = pd.DataFrame([values])
        pred, proba, ok, err = Pred_func(row, o, model_name)
        if cfg["task"] == "classification":
            st.success(f"{cfg['target']} prédit : **{pred[0]}**")
            if proba is not None:
                st.caption("Probabilité estimée pour chaque classe")
                st.bar_chart(proba.iloc[0])
        else:
            st.success(f"{cfg['target']} prédit : **{fmt(float(pred[0]))}{unit_of(o)}**")
        for f in o["features"]:
            if f["kind"] == "num" and f["min"] is not None and not f["min"] <= values[f["name"]] <= f["max"]:
                st.warning(f"{f['name']} est hors de la plage d'entraînement "
                           f"[{f['min']:g} ; {f['max']:g}] : prédiction peu fiable.")
        if len(o["models"]) > 1:
            with st.expander("Comparer avec les autres modèles"):
                rows = []
                for m in o["models"]:
                    p = Pred_func(row, o, m, with_proba=False)[0][0]
                    rows.append({"Modèle": m, f"{cfg['target']} prédit": p if cfg["task"] == "classification"
                                 else fmt(float(p)) + unit_of(o)})
                st.table(pd.DataFrame(rows))


# ----------------------------------------------------------------------------------
# Interface 2 : prédiction multiple (fichier)
# ----------------------------------------------------------------------------------
def template_df(o):
    row = {}
    for f in o["features"]:
        row[f["name"]] = (o["encoders"][f["name"]].classes_[0] if f["kind"] == "cat"
                          else (int(round(f["median"])) if f["is_int"] else round(f["median"], 2)))
    return pd.DataFrame([row])


def multiple_interface(ds, o, model_name):
    cfg = o["cfg"]
    st.write("Chargez un fichier avec **une ligne par observation** : une colonne de prédiction est ajoutée.")
    st.write("Colonnes requises (ordre libre, accents/majuscules ignorés) : "
             + ", ".join(f"`{f['name']}`" for f in o["features"]))
    st.download_button("Télécharger un fichier exemple (csv)",
                       data=template_df(o).to_csv(index=False).encode("utf-8"),
                       file_name=f"exemple_{ds}.csv", mime="text/csv", key=f"ex_{ds}_{model_name}")

    file = st.file_uploader("Charger un fichier (csv, tsv, txt, xlsx, json)", type=FILE_TYPES,
                            key=f"up_{ds}_{model_name}")
    if file is None:
        return
    try:
        df = read_file(file)
    except Exception as e:
        st.error(f"Lecture impossible : {e}")
        return
    st.write(f"Aperçu des données chargées ({len(df)} lignes) :")
    st.dataframe(df.head())

    state, file_id = f"res_{ds}_{model_name}", f"{file.name}:{file.size}"
    if st.button("Prédire", key=f"btn_{ds}_{model_name}"):
        try:
            st.session_state[state] = (file_id, *Pred_func_csv(df, o, model_name))
        except ValueError as e:
            st.session_state.pop(state, None)
            st.error(str(e))

    res = st.session_state.get(state)
    if res and res[0] == file_id:
        _, out, ok = res
        st.success(f"{int(ok.sum())} ligne(s) prédite(s) sur {len(out)}.")
        if not ok.all():
            st.warning(f"{int((~ok).sum())} ligne(s) ignorée(s) : voir la colonne « erreur ».")
        st.write("Résultats des prédictions :")
        st.dataframe(out)
        st.download_button("Télécharger les prédictions (csv)",
                           data=out.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"predictions_{ds}.csv", mime="text/csv", key=f"dl_{ds}_{model_name}")


# ----------------------------------------------------------------------------------
# Page principale
# ----------------------------------------------------------------------------------
st.title("🤖 Machine Learning - Prédictions")
available = [k for k, c in DATASETS.items() if not missing_files(c)]
for k, c in DATASETS.items():
    if k not in available:
        st.sidebar.caption(f"⚠️ {c['title']} : fichier(s) manquant(s) dans `models/` → "
                           + ", ".join(missing_files(c)))
if not available:
    st.error("Aucun modèle trouvé. Placez dans le dossier `models/` les fichiers `best_model*.joblib`, "
             "`encoders*.joblib`, `scaler*.joblib` et `uniques*.joblib` générés par les notebooks "
             "(suffixes : aucun = Car_Data, `_dataset` = Car_dataset, `_tel` = Téléphones).")
    st.stop()

ds = st.sidebar.radio("Jeu de données", available, format_func=lambda k: DATASETS[k]["title"])
o, problems = load_objects(ds)
if problems:
    st.error("Les fichiers chargés ne correspondent pas à la configuration de `app.py` :")
    for p in problems:
        st.write("• " + p)
    st.stop()

cfg = o["cfg"]
st.sidebar.markdown(f"**Cible :** {cfg['target']}  \n**Type :** "
                    f"{'classification' if cfg['task'] == 'classification' else 'régression'}  \n"
                    f"**Variables :** {', '.join(f['name'] for f in o['features'])}")
st.header(cfg["title"])

names = list(o["models"])
tabs = st.tabs(names)
for tab, m in zip(tabs, names):
    with tab:
        tab_simple, tab_multi = st.tabs(["🔹 Prédiction simple", "📂 Prédiction multiple"])
        with tab_simple:
            simple_interface(ds, o, m)
        with tab_multi:
            multiple_interface(ds, o, m)
