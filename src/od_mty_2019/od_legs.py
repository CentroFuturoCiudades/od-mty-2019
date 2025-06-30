"""Changes the leg table from wide to long format.
Fixes issued with leg data."""

import numpy as np
import pandas as pd


def build_legs(legs_wide):
    """Changes the leg table from wide to long format.
    Fixes issued with leg data."""

    m1_cols = [
        "M1_Transp",
        "M1_TipoTransp",
        "M1_Transp_O",
        "M1Tpo_Caminata",
        "M1N_Ruta",
        "M1_HHTpoParada",
        "M1_MMTpoParada",
        "M1_HHTpoAbordo",
        "M1_HHTpoAbordo_O",
        "M1_MMTpoAbordo",
        "M1_Pago",
    ]
    m2_cols = [
        "M2_Transp",
        "M2_TpoTranspordo",
        "M2_TipoTransp",
        "M2_Transp_O",
        "M2Tpo_Caminata",
        "M2N_Ruta",
        "M2_HHTpoParada",
        "M2_MMTpoParada",
        "M2_HHTpoAbordo",
        "M2_HHTpoAbordo_O",
        "M2_MMTpoAbordo",
        "M2_Pago",
    ]
    m3_cols = [
        "M3_Transp",
        "M3_TpoTranspordo",
        "M3_TipoTransp",
        "M3_Transp_O",
        "M3Tpo_Caminata",
        "M3N_Ruta",
        "M3_HHTpoParada",
        "M3_MMTpoParada",
        "M3_HHTpoAbordo",
        "M3_HHTpoAbordo_O",
        "M3_MMTpoAbordo",
        "M3_Pago",
    ]
    m4_cols = [
        "M4_Transp",
        "M4_TpoTranspordo",
        "M4_TipoTransp",
        "M4_Transp_O",
        "M4Tpo_Caminata",
        "M4N_Ruta",
        "M4_HHTpoParada",
        "M4_MMTpoParada",
        "M4_HHTpoAbordo",
        "M4_HHTpoAbordo_O",
        "M4_MMTpoAbordo",
        "M4_Pago",
    ]
    m5_cols = [
        "M5_Transp",
        "M5_TpoTranspordo",
        "M5_TipoTransp",
        "M5_Transp_O",
        "M5Tpo_Caminata",
        "M5N_Ruta",
        "M5_HHTpoParada",
        "M5_MMTpoParada",
        "M5_HHTpoAbordo",
        "M5_HHTpoAbordo_O",
        "M5_MMTpoAbordo",
        "M5_Pago",
    ]
    m6_cols = [
        "M6_Transp",
        "M6_TpoTranspordo",
        "M6_TipoTransp",
        "M6_Transp_O",
        "M6Tpo_Caminata",
        "M6N_Ruta",
        "M6_HHTpoParada",
        "M6_MMTpoParada",
        "M6_HHTpoAbordo",
        "M6_HHTpoAbordo_O",
        "M6_MMTpoAbordo",
        "M6_Pago",
    ]

    legs1 = legs_wide[m1_cols].rename(columns=lambda c: c.replace("M1", "").strip("_"))
    legs2 = legs_wide[m2_cols].rename(columns=lambda c: c.replace("M2", "").strip("_"))
    legs3 = legs_wide[m3_cols].rename(columns=lambda c: c.replace("M3", "").strip("_"))
    legs4 = legs_wide[m4_cols].rename(columns=lambda c: c.replace("M4", "").strip("_"))
    legs5 = legs_wide[m5_cols].rename(columns=lambda c: c.replace("M5", "").strip("_"))
    legs6 = legs_wide[m6_cols].rename(columns=lambda c: c.replace("M6", "").strip("_"))

    nan_vals = [
        0,
        False,
        "0",
        "no utilizó otro modo de transporte",
        "no utilizó otro modo de transporte",
        "no utilizo otro medio de transporte",
        "no utilizó otro medio de transporte",
    ]
    for legs in [legs1, legs2, legs3, legs4, legs5, legs6]:
        legs.loc[:, "TipoTransp"] = (
            legs.loc[:, "TipoTransp"]
            .str.strip()
            .str.lower()
            .str.normalize("NFKD")
            .replace(nan_vals, np.nan)
            .replace("transporte público", "público")
            .replace("vehículo particular", "particular")
            .replace("a pie (caminando)", "caminó")
            .replace("transpote por aplicación", "transporte por aplicación")
        )

        legs.loc[:, "Transp"] = (
            legs.loc[:, "Transp"]
            .str.strip()
            .str.lower()
            .str.normalize("NFKD")
            .replace(nan_vals, np.nan)
            .replace("autobús  suburbano", "autobús suburbano")
            .replace("uber, cabify , didi o similar", "uber, cabify, didi, o similar")
        )
        legs["TpoAbordo"] = (
            legs[["HHTpoAbordo", "HHTpoAbordo_O"]].max(axis=1) * 60 + legs.MMTpoAbordo
        )
        legs["TpoParada"] = legs.HHTpoParada * 60 + legs.MMTpoParada
        legs.drop(
            columns=[
                "HHTpoAbordo",
                "HHTpoAbordo_O",
                "MMTpoAbordo",
                "HHTpoParada",
                "MMTpoParada",
            ],
            inplace=True,
        )

    legs6 = legs6.dropna(subset="Transp")
    legs5 = legs5.dropna(subset="Transp")
    legs4 = legs4.dropna(subset="Transp")
    # Legs 4 have mislabeled walking legs
    legs4.loc[
        (legs4.Transp == "a pie (caminando)") & legs4.TipoTransp.isna(), "TipoTransp"
    ] = "caminó"
    # Legs 3 has a mislabled leg
    legs3.loc[legs3.Transp.notnull() & legs3.TipoTransp.isna(), "TipoTransp"] = "caminó"
    legs3 = legs3.dropna(subset="Transp")
    legs2 = legs2.dropna(subset="Transp")
    legs2.loc[("59928-4", 2, 2), "TipoTransp"] = "otro modo"
    legs2 = legs2.dropna(subset="TipoTransp")
    legs2 = legs2.drop(index=("35090-30", 2, 2))

    # For Legs1, many legs seem
    # to mix walk first leg to
    # another mode, reported in TipoTransp
    cond = (
        (legs1.Transp == "a pie (caminando)")
        & (legs1.TipoTransp == "otro modo")
        & (legs1.TpoAbordo == 0)
    )
    legs1.loc[cond, "TipoTransp"] = "caminó"

    cond = (
        (legs1.Transp == "a pie (caminando)")
        & (legs1.TipoTransp == "otro modo")
        & (legs1.TpoAbordo > 0)
    )
    legs1.loc[cond, "Transp"] = "otro"

    # return legs1, legs2, legs3, legs4, legs5, legs6

    # nnans = legs6.replace(nan_cols, np.nan).T.isna().sum()
    # legs6 = legs6[(nnans < 12)].copy()
    # return legs6

    # nnans = legs5.replace(nan_cols, np.nan).T.isna().sum()
    # legs5 = legs5[(nnans < 11)].copy()

    # nnans = legs4.replace(nan_cols, np.nan).T.isna().sum()
    # legs4 = legs4[(nnans < 12)].copy()

    # nnans = legs3.replace(nan_cols, np.nan).T.isna().sum()
    # legs3 = legs3[(nnans < 11)].copy()

    # legs2 = legs2.query(
    #     "Transp != 'No utilizó otro medio de transporte'"
    # ).copy()

    legs1 = legs1.assign(TRAMO=1).set_index("TRAMO", append=True)
    legs2 = legs2.assign(TRAMO=2).set_index("TRAMO", append=True)
    legs3 = legs3.assign(TRAMO=3).set_index("TRAMO", append=True)
    legs4 = legs4.assign(TRAMO=4).set_index("TRAMO", append=True)
    legs5 = legs5.assign(TRAMO=5).set_index("TRAMO", append=True)
    legs6 = legs6.assign(TRAMO=6).set_index("TRAMO", append=True)

    legs = pd.concat([legs2, legs3, legs4, legs5, legs6, legs1], axis=0).sort_index()

    legs["Transp"] = (
        legs.Transp.str.strip()
        .replace(
            [
                "taxi",
                "Caminó",
                "Autobús foráneo",
                "Automóvil (Pasajero)",
                "Automóvil (conductor)",
                "Metro Enlace",
                "Automóvil\xa0(Conductor)",
                "Automóvil\xa0(pasajero)",
                "Motocicleta (conductor)",
                "Uber, Cabify, Didi, o similar",
                "Autobús  Suburbano",
                "Transporte de personal",
            ],
            [
                "Taxi",
                "A pie (caminando)",
                "Autobús Foráneo",
                "Automóvil (pasajero)",
                "Automóvil (Conductor)",
                "Metro enlace",
                "Automóvil (Conductor)",
                "Automóvil (pasajero)",
                "Motocicleta (Conductor)",
                "Uber, Cabify , Didi o similar",
                "Autobús Suburbano",
                "Transporte de Personal",
            ],
        )
        .replace(
            [
                "A pie (caminando)",
                "Uber, Cabify , Didi o similar",
                "Automóvil (Conductor)",
                "Automóvil (pasajero)",
                "Autobús Suburbano",
                "Camión Urbano",
                "Ecovía",
                "Metrobús",
                "Metrorrey",
                "Microbús",
                "Transmetro",
                "Transporte Público",
                "Metro enlace",
                "Motocicleta (Conductor)",
                "Motocicleta (pasajero)",
                "Bicicleta",
                "Autobús Foráneo",
                "Otro",
            ],
            [
                "caminando",
                "app",
                "auto",
                "auto",
                "TPUB",
                "TPUB",
                "TPUB",
                "TPUB",
                "TPUB",
                "TPUB",
                "TPUB",
                "TPUB",
                "TPUB",
                "moto",
                "moto",
                "bici",
                "otro",
                "otro",
            ],
        )
    )

    return legs
