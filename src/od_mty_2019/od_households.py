"""Generate the households table from a clean OD dataframe."""

import numpy as np


def build_household_table(od_df, people):
    """Builds the household table from the od survey. Cleans up in the process."""

    viv_cols = [
        "MUN",
        "TAZ",
        "LineaTelef",
        "Internet",
        "VHAuto",
        "VHMoto",
        "VHPickup",
        "VHCamion",
        "VHBici",
        "VHPatineta",
        "VHPatines",
        "VHScooter",
        "VHOtro",
        "CHBaños",
        "CHDormitorios",
        "Hab14masTrabajo",
        "HabitantesTotal",
        "HbitantesMayor6",
        "HbitantesMenor5",
        "HabitantesObs",
        "NSE",
        "NIntDom",
    ]

    viv_df = od_df[viv_cols].groupby("HOGAR").first()

    viv_df["NumberOfVehicles"] = viv_df.VHAuto + viv_df.VHPickup + viv_df.VHMoto

    # Auxiliary columns to fix household counts
    viv_df["HabitantesObs"] = people.groupby(["HOGAR"]).size()
    viv_df["Hab14masTrabajoObs"] = (
        people.query("Edad >= 14 & CONACT == 'Trabajó'").groupby(["HOGAR"]).size()
    )
    viv_df["Hab14masTrabajoObs"] = viv_df["Hab14masTrabajoObs"].fillna(0.0)
    viv_df["Hab14masNTrabajoObs"] = (
        people.query("Edad >= 14 & CONACT != 'Trabajó'").groupby(["HOGAR"]).size()
    )
    viv_df["Hab14masNTrabajoObs"] = viv_df["Hab14masNTrabajoObs"].fillna(0.0)

    # Fix counts
    viv_df_2 = viv_df.copy()

    viv_df_2["Hab14masTrabajo"] = viv_df_2.Hab14masTrabajo.mask(
        viv_df.Hab14masTrabajo < viv_df.Hab14masTrabajoObs, viv_df.Hab14masTrabajoObs
    )

    viv_df_2["HbitantesMayor6"] = viv_df_2.HbitantesMayor6.mask(
        viv_df_2.HabitantesObs > viv_df_2.HbitantesMayor6, viv_df_2.HabitantesObs
    )

    viv_df_2["HabitantesTotal"] = viv_df_2.HbitantesMenor5 + viv_df_2.HbitantesMayor6

    viv_df_2["Hab14masTrabajoSup"] = (
        viv_df_2.HbitantesMayor6 - viv_df_2.Hab14masNTrabajoObs
    )
    viv_df_2["Hab14masTrabajo"] = viv_df_2.Hab14masTrabajo.mask(
        viv_df_2.Hab14masTrabajo > viv_df_2.Hab14masTrabajoSup,
        viv_df_2.Hab14masTrabajoSup,
    )

    perc_adj = (
        ((viv_df_2.drop(columns="Hab14masTrabajoSup") != viv_df).sum(axis=1) > 0).sum()
        / len(viv_df)
        * 100
    )
    print(f"{perc_adj}% of households have been ajusted.")

    perc_incom = (
        (viv_df_2.HabitantesObs != viv_df_2.HbitantesMayor6).sum() / len(viv_df) * 100
    )
    print(f"{perc_incom} of households are missing members " "above 6 years of age.")

    viv_df = viv_df_2.drop(columns=["Hab14masTrabajoSup", "Hab14masNTrabajoObs"])

    viv_df = viv_df.rename(columns={"LineaTelef": "TELEFONO", "Internet": "INTERNET"})
    viv_df["AUTOPROP"] = (
        (viv_df.VHAuto + viv_df.VHPickup).astype(bool).map({True: "Sí", False: "No"})
    )
    viv_df["MOTOCICLETA"] = viv_df.VHMoto.astype(bool).map({True: "Sí", False: "No"})
    viv_df["BICICLETA"] = viv_df.VHBici.astype(bool).map({True: "Sí", False: "No"})
    viv_df["NUMPERS"] = viv_df.HabitantesTotal.astype(int)
    viv_df["CUADORM"] = viv_df.CHDormitorios.astype(int)

    # Augment columns
    aug_cols = ["EDUC", "EDAD", "ACTIVIDADES_C", "ASISTEN", "CONACT", "SITTRA"]
    viv_df[aug_cols] = (
        people.query("PARENTESCO == 'Jefa(e)'")[aug_cols]
        .droplevel(1)
        .reindex(viv_df.index)
    )

    viv_df = viv_df.replace("Si", "Sí")

    # Drop the 4 households withou family head
    # viv_df = viv_df.drop(['22899-4', '25131-4', '3275-118', '59886-12'])

    viv_df = viv_df.fillna(value=np.nan)

    # Dwelling type will be infered from dwellings with
    # inner adress number.
    # Just two classes
    viv_df["CLAVIVP"] = (
        viv_df.NIntDom.str.strip()
        .str.lower()
        .replace(["0", "s/n", "", "s7n", "n/p", "n/r", "nt"], None)
        .notnull()
        .map({True: "ConNumInt", False: "SinNumInt"})
    )
    viv_df = viv_df.drop(columns="NIntDom")

    return viv_df
