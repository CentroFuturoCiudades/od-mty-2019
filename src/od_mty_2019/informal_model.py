"""Trains the model to impute formal/informal job trip status."""

from pickle import dump, load

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def od_to_enoe(people):
    """Process cleaned OD survey into expected model shape."""

    od_model = (
        people.query("CONACT=='Trabajó'")[
            ["SEXO", "SITTRA", "Edad", "ACTIVIDADES_C", "EDUC2", "MUN"]
        ]
        .copy()
        .pipe(
            lambda df: df.assign(
                genero=df.SEXO.map({"M": "H", "F": "F"}),
                ocupacion=df.SITTRA.map(
                    {
                        "empleada(o) u obrera(o)": "trabajador",
                        "trabajador(a) por cuenta propia": "independiente",
                        "patrón(a) o empleador(a)": "otro",
                    }
                ).fillna("otro"),
                edad_num=df.Edad,
                sector=df.ACTIVIDADES_C.fillna("otro"),
                escolaridad=df.EDUC2.fillna("Básica"),
                municipio=df.MUN.str.lower().replace(
                    [
                        "cadereyta jiménez",
                        "abasolo",
                        "el carmen",
                        "general escobedo",
                        "ciénega de flores",
                        "garcía",
                        "general zuazua",
                        "juárez",
                        "pesquería",
                        "salinas victoria",
                        "san nicolás de los garza",
                        "santa catarina",
                        "san pedro garza garcía",
                    ],
                    [
                        "cadereyta",
                        "otro",
                        "carmen",
                        "escobedo",
                        "flores",
                        "garcia",
                        "zuazua",
                        "juarez",
                        "otro",
                        "salinas",
                        "san_nicolas",
                        "santa_catarina",
                        "san_pedro",
                    ],
                ),
            )
        )
        .drop(columns=["SEXO", "SITTRA", "Edad", "ACTIVIDADES_C", "EDUC2", "MUN"])
    )
    return od_model


def train_model():
    """Trains informal/formal job classification model on enoe data.
    Saves model as a pickle file.
    """

    enoe = pd.read_csv("data/enoe_clean.csv", index_col=0)
    y = enoe["informal"]
    X = enoe.drop(columns="informal").pipe(
        lambda df: df.assign(
            sector=df.sector.str.lower(),
            escolaridad=df.escolaridad.map(
                {
                    "Primaria o Secundaria": "Básica",
                    "Carrera técnica o preparatoria": "MediaSup",
                    "Sin Instrucción": "Sin Educación",
                    "Licenciatura": "Licenciatura",
                    "Postgrado": "Postgrado",
                }
            ),
        )
    )

    cat_cols_no_edad = [
        "genero",
        "ocupacion",
        "sector",
        "escolaridad",
        "municipio",
    ]

    model = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "od_encoder",
                            OrdinalEncoder(
                                handle_unknown="error",
                                unknown_value=None,
                            ),
                            cat_cols_no_edad,
                        ),
                        ("passthrough", "passthrough", ["edad_num"]),
                    ]
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=1000,
                    early_stopping=True,
                    random_state=0,
                    learning_rate=0.1,
                    max_leaf_nodes=5,
                ),
            ),
        ]
    )

    model.fit(X, y)

    with open("data/outputs/informal_model.pkl", "wb") as f:
        dump(model, f, protocol=5)


def classify_job(people):
    """Classify trip as formal or informal.
    Return DataFrame with added column.
    """

    # Avoid modifying inputs
    people = people.copy()
    with open("data/outputs/informal_model.pkl", "rb") as f:
        model = load(f)

    # Process cleaned OD survey into expected model shape
    od_model = od_to_enoe(people)

    # Predict formal/informal labels
    od_model["informal"] = model.predict(od_model)

    people["informal"] = (
        od_model["informal"].reindex(people.index).fillna("Blanco por pase")
    )

    return people
