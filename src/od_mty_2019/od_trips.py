"""Functions that deal with trip level data in the OD survey.

Attempt to fix issues with trips, such as start and ending times and inconsistent
trip sequence.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def plot_trips(h, df):
    """Utility function to plot all trips from a household (h)."""

    df = df.loc[h].reset_index()
    motivos_map = {
        "acompañar / recoger": 0,
        "compras": 1,
        "estudios": 2,
        "otro": 5,
        "recreación": 4,
        "regreso a casa": 3,
        "salud": 6,
        "trabajo": 7,
    }

    cmap = mpl.colormaps["tab10"]

    custom_lines = [Line2D([0], [0], color=cmap(v), lw=4) for v in motivos_map.values()]

    max_hab = df.HABITANTE.max()

    plt.figure(figsize=(14, max_hab / 2))

    for fecha in df.fecha_inicio.unique():
        plt.axvline(fecha, color="grey", ls="--", lw=1)
    for fecha in df.fecha_termino.unique():
        plt.axvline(fecha, color="grey", ls="--", lw=1)
    plt.hlines(
        df.HABITANTE,
        df.fecha_inicio,
        df.fecha_termino,
        lw=20,
        colors=cmap(df.Motivo.map(motivos_map)),
    )
    plt.xticks(rotation=0)
    df_h = df.groupby("HABITANTE").first().reset_index()
    plt.yticks(
        df_h.HABITANTE,
        [
            f"{r.Género[0]}{r.Edad} {r.Ocupacion}, "
            f"{r.RelaciónHogar} ({r.HABITANTE})"
            for i, r in df_h.iterrows()
        ],
    )
    plt.ylim(0, df.HABITANTE.max() + 1)
    plt.legend(custom_lines, motivos_map.keys(), bbox_to_anchor=(1.01, 1))


def insert_trip(
    df, hogar, habitante, trip_num, motivo, modo, fecha_inicio, fecha_termino
):
    """Inserts a trip between existing trips.
    Automatically sets Origin Destination from prev and next trip.
    Must provide fecha inicio and termino, motivo y modo.
    """
    df = df.copy()

    df_hab = df.loc[(hogar, habitante)]

    # Find current trip number.
    new_trip = df.loc[(hogar, habitante, trip_num)].copy()
    new_trip.loc[["Lugar_Or", "LugarDest"]] = np.nan
    new_trip.loc[["ZonaOri"]] = df_hab.loc[trip_num - 1, "ZonaDest"]
    new_trip.loc[["ZonaDest"]] = df_hab.loc[trip_num, "ZonaOri"]
    new_trip.loc[["Origen"]] = df_hab.loc[trip_num - 1, "Destino"]
    new_trip.loc[["Destino"]] = df_hab.loc[trip_num, "Origen"]
    new_trip.loc[["Motivo", "Modo Agrupado", "fecha_inicio", "fecha_termino"]] = [
        motivo,
        modo,
        fecha_inicio,
        fecha_termino,
    ]
    new_trip.loc["duracion"] = new_trip.fecha_termino - new_trip.fecha_inicio

    # Renumber viajes
    df = df.reset_index()
    cond = (df.HOGAR == hogar) & (df.HABITANTE == habitante) & (df.VIAJE >= trip_num)
    df.loc[cond, "VIAJE"] = df.loc[cond, "VIAJE"] + 1
    df = df.set_index(["HOGAR", "HABITANTE", "VIAJE"])

    df.loc[(hogar, habitante, trip_num)] = new_trip

    return df.sort_index()


def fix_od_chains(trips):
    """Makes sure next trip origin is previous trip destination.
    Changes are made in place.
    """

    trips.loc[("1020-26", 2, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        431.0,
        431.0,
        "Otro",
        "Hogar",
    ]

    trips.loc[("134-2-009-34", 1, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        620,
        620,
        "Otro",
        "Otro",
    ]

    trips.loc[("1342012-26", 2, 3), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        617,
        206,
        "Hogar",
        "Otro",
    ]
    trips.loc[("1342012-26", 2, 6), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        206,
        203,
        "Otro",
        "Otro",
    ]

    trips.loc[("1342012-8", 2, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        206,
        214,
        "Otro",
        "Otro",
    ]

    trips.loc[("14493-6", 1, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        404,
        634,
        "Hogar",
        "Otro",
    ]
    trips.loc[("14493-6", 1, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        634,
        404,
        "Otro",
        "Hogar",
    ]

    trips.loc[("14525-10", 1, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        406,
        81,
        "Hogar",
        "Otro",
    ]
    trips.loc[("14525-10", 1, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        81,
        406,
        "Otro",
        "Hogar",
    ]

    trips.loc[("16167-20", 1, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        421,
        241,
        "Hogar",
        "Otro",
    ]
    trips.loc[("16167-20", 1, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        241,
        421,
        "Otro",
        "Hogar",
    ]

    trips.loc[
        ("17841-14", 2, 4), ["ZonaOri", "ZonaDest", "Origen", "Destino", "Motivo"]
    ] = [426, 426, "Otro", "Hogar", "regreso a casa"]

    trips.loc[("17863-12", 1, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        429,
        372,
        "Hogar",
        "Otro",
    ]
    trips.loc[("17863-12", 1, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        372,
        429,
        "Otro",
        "Hogar",
    ]

    trips.loc[("17889-16", 3, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        428,
        4,
        "Hogar",
        "Otro",
    ]
    trips.loc[("17889-16", 3, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        4,
        428,
        "Otro",
        "Hogar",
    ]

    trips.loc[("17927-12", 1, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        428,
        238,
        "Hogar",
        "Otro",
    ]
    trips.loc[("17927-12", 1, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        238,
        238,
        "Otro",
        "Hogar",
    ]

    trips.loc[("181-118", 5, 2), ["Origen"]] = ["Escuela"]

    trips.loc[("18322-20", 3, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        335,
        221,
        "Hogar",
        "Otro",
    ]
    trips.loc[("18322-20", 3, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        221,
        335,
        "Otro",
        "Hogar",
    ]

    trips.loc[("19880-8", 3, 4), ["fecha_inicio", "fecha_termino"]] = [
        pd.to_datetime("2019-09-18 17:40:00"),  # +00:00'),
        pd.to_datetime("2019-09-18 17:50:00"),  # +00:00')
    ]

    # WARNING: This changes trip id here, but not in the tiplegs table.
    # FIXME. TODO.
    # trips = insert_trip(
    #             trips, '19880-8', 2, 4, 'regreso a casa', 'caminando',
    #             pd.to_datetime('2019-09-18 17:40:00+00:00'),
    #             pd.to_datetime('2019-09-18 17:50:00+00:00'),
    #            )

    # trips.loc[
    #     ('25161A-4', 1, 2), ['fecha_inicio', 'fecha_termino']
    # ] = trips.loc[
    #     ('25161A-4', 1, 2), ['fecha_inicio', 'fecha_termino']
    # ] + pd.to_timedelta('4 days')
    # trips.loc[
    #     ('25161A-4', 4, 2), ['fecha_inicio', 'fecha_termino']
    # ] = trips.loc[
    #     ('25161A-4', 4, 2), ['fecha_inicio', 'fecha_termino']
    # ] + pd.to_timedelta('4 days')

    trips.loc[
        ("25161A-4", 3, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino", "Motivo"]
    ] = [282, 282, "Otro", "Hogar", "regreso a casa"]
    trips.loc[("28755-4", 4, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        68,
        204,
        "Hogar",
        "Otro",
    ]
    trips.loc[("28755-4", 4, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        204,
        68,
        "Otro",
        "Hogar",
    ]

    trips.loc[("34518-6", 2, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        2,
        5,
        "Otro",
        "Hogar",
    ]

    trips.loc["353-201", ["ZonaOri", "ZonaDest"]] = (
        trips.loc["353-201", ["ZonaOri", "ZonaDest"]].replace(620, 621).values
    )
    trips.loc[("353-201", 4, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        1,
        621,
        "Tienda/(Super)mercado",
        "Hogar",
    ]

    trips.loc[("35709-6", 2, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        239,
        221,
        "Hogar",
        "Otro",
    ]
    trips.loc[("35709-6", 2, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        221,
        239,
        "Otro",
        "Hogar",
    ]

    trips.loc[("42188-18", 2, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        237,
        221,
        "Hogar",
        "Otro",
    ]
    trips.loc[("42188-18", 2, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        221,
        237,
        "Otro",
        "Hogar",
    ]

    trips.loc[("42192-16", 3, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        239,
        754,
        "Hogar",
        "Otro",
    ]
    trips.loc[("42192-16", 3, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        754,
        239,
        "Otro",
        "Hogar",
    ]

    trips.loc[("42192-22", 1, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        239,
        706,
        "Hogar",
        "Otro",
    ]
    trips.loc[("42192-22", 1, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        706,
        239,
        "Otro",
        "Hogar",
    ]

    trips.loc[("42211-34", 3, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        239,
        706,
        "Hogar",
        "Otro",
    ]
    trips.loc[("42211-34", 3, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        706,
        239,
        "Otro",
        "Hogar",
    ]

    trips.loc[("45501-6", 2, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        770,
        606,
        "Hogar",
        "Otro",
    ]
    trips.loc[("45501-6", 2, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        606,
        770,
        "Otro",
        "Hogar",
    ]

    trips.loc[("58995-4", 4, 1), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        446,
        715,
        "Hogar",
        "Otro",
    ]
    trips.loc[("58995-4", 4, 2), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        715,
        446,
        "Otro",
        "Hogar",
    ]

    trips.loc["6043-403", ["ZonaOri", "ZonaDest"]] = (
        trips.loc["6043-403", ["ZonaOri", "ZonaDest"]].replace(107, 108).values
    )
    trips.loc[("6043-403", 1, 2), "Origen"] = "Recreativo"
    trips.loc[("6043-403", 2, 2), "Origen"] = "Recreativo"

    trips.loc[("9681-5", 2, 3), ["ZonaOri", "ZonaDest", "Origen", "Destino"]] = [
        207,
        87,
        "Otro",
        "Otro",
    ]


def check_od_chains(trips):
    """Validates trips origin and destination chain properly.
    Returns household, people whose trips fail to chain."""

    # Check in prev destination is current origin
    # current orgin starts at trips > 2
    idxs = pd.IndexSlice
    cur_idx = trips.loc[idxs[:, :, 2:]].index
    prev_idx = pd.MultiIndex.from_arrays(
        [
            cur_idx.get_level_values(0),
            cur_idx.get_level_values(1),
            cur_idx.get_level_values(2) - 1,
        ]
    )

    no_chain = (trips.loc[cur_idx].Origen != trips.loc[prev_idx].Destino.values).pipe(
        lambda s: s[s].index.droplevel(2).unique()
    )

    return no_chain


def check_overlap(df):
    """Validates if next trips start time is greater than previous trips ending time."""

    idxs = pd.IndexSlice
    # Index for current trips (trip number > 2)
    cur_idx = df.loc[idxs[:, :, 2:]].index
    # index for previous trips
    prev_idx = pd.MultiIndex.from_arrays(
        [
            cur_idx.get_level_values(0),
            cur_idx.get_level_values(1),
            cur_idx.get_level_values(2) - 1,
        ]
    )

    overlaps = (
        df.loc[cur_idx].fecha_inicio.values < df.loc[prev_idx].fecha_termino.values
    )

    return cur_idx[overlaps]  # .droplevel(2).unique()


def get_purpose_tmat(trips, ignore=None):
    """Builds the purpose transition matrix."""

    if ignore is not None:
        trips = trips.drop(index=ignore)
    trips = trips.copy()

    idxs = pd.IndexSlice
    cur_idx = trips.loc[idxs[:, :, 2:]].index
    prev_idx = pd.MultiIndex.from_arrays(
        [
            cur_idx.get_level_values(0),
            cur_idx.get_level_values(1),
            cur_idx.get_level_values(2) - 1,
        ]
    )

    return pd.crosstab(
        trips.loc[prev_idx].Motivo.values, trips.loc[cur_idx].Motivo.values
    )


def index_next_trip(idx):
    """Builds multi index of next trip."""

    nidx = pd.MultiIndex.from_arrays(
        [idx.get_level_values(0), idx.get_level_values(1), idx.get_level_values(2) + 1]
    )

    return nidx


def index_prev_trip(idx):
    """Builds multi index of previous trip."""

    nidx = pd.MultiIndex.from_arrays(
        [idx.get_level_values(0), idx.get_level_values(1), idx.get_level_values(2) - 1]
    )

    return nidx


def fix_home_loc(trips, hogar, habitante):
    """Fix zone codes for stated home orgin or destinations.
    Imputes code for home as the household zone."""

    idxs = pd.IndexSlice
    taz = trips.loc[(hogar, habitante, 1)].TAZ

    # Find wrong home from first trip
    if trips.loc[(hogar, habitante, 1), "Origen"] == "Hogar":
        wrong_taz = trips.loc[(hogar, habitante, 1), "ZonaOri"]
    elif trips.loc[(hogar, habitante, 1), "Destino"] == "Hogar":
        wrong_taz = trips.loc[(hogar, habitante, 1), "ZonaDest"]
    else:
        assert False, (hogar, habitante)

    # Replace all instances with true TAZ
    # WARNING. This assumes all zones with wrong taz are actually taz
    # This may not be true, and trips amons tazs may indeed ocurr.
    trips.loc[(hogar, habitante), ["ZonaOri", "ZonaDest"]] = (
        trips.loc[(hogar, habitante), ["ZonaOri", "ZonaDest"]]
        .replace(wrong_taz, taz)
        .values
    )

    # Replace Origen Destino home with true TAZ
    idx = trips.loc[(hogar, habitante)].query("Origen == 'Hogar'").index
    trips.loc[idxs[hogar, habitante, idx], "ZonaOri"] = taz
    idx = trips.loc[(hogar, habitante)].query("Destino == 'Hogar'").index
    trips.loc[idxs[hogar, habitante, idx], "ZonaDest"] = taz

    # Fix chains by backprogating next trip Origin to previous trip Destination
    # only if destination is not home, if home then propagate forward.
    # Iterating over all trips of the inhabitant
    # These are typically wrong by duplicating origen destino in the row
    # If this is the case, we keep the home as the true value and change the
    # other
    last_trip = trips.loc[(hogar, habitante, 1)]["last"]
    # p_zorigen = trips.loc[(hogar, habitante, 1)].ZonaOri
    p_zdestino = trips.loc[(hogar, habitante, 1)].ZonaDest
    # p_origen = trips.loc[(hogar, habitante, 1)].Origen
    p_destino = trips.loc[(hogar, habitante, 1)].Destino
    for i in range(2, last_trip + 1):
        zorigen = trips.loc[(hogar, habitante, i)].ZonaOri
        # zdestino = trips.loc[(hogar, habitante, i)].ZonaDest
        origen = trips.loc[(hogar, habitante, i)].Origen
        # destino = trips.loc[(hogar, habitante, i)].Destino
        if zorigen != p_zdestino:
            # There is a chain problem
            # Is either home?
            assert origen != "Hogar"
            assert p_destino != "Hogar"

            # Is one is TAZ an the other is different from TAZ
            # keep the different one
            if zorigen != taz:
                z = zorigen
            elif p_zdestino != taz:
                z = p_zdestino
            else:
                assert False, "Not implemented."
            # Make the change
            trips.loc[(hogar, habitante, i - 1), "ZonaDest"] = z
            trips.loc[(hogar, habitante, i), "ZonaOri"] = z

        # Update vars
        # p_zorigen = trips.loc[(hogar, habitante, i)].ZonaOri
        p_zdestino = trips.loc[(hogar, habitante, i)].ZonaDest
        # p_origen = trips.loc[(hogar, habitante, i)].Origen
        p_destino = trips.loc[(hogar, habitante, i)].Destino


def build_trips(od_df):
    """Builds the trips table and the legs table from a clean od data frame.
    Further adjusts trips information to obtain a self consistent trip table."""

    # Get trip table
    trip_cols = [
        "TAZ",
        # 'HabitantesTotal',
        # 'Edad', 'Género', 'RelaciónHogar', 'Estudios',
        "Ocupacion",  # 'Ocupacion_O', 'SectorEconom', 'SectorEconom_O',
        "Lugar_Or",
        "LugarDest",
        # 'Macrozona Origen', 'Macrozona Destino',
        "ZonaOri",
        "ZonaDest",
        "Origen",
        "Destino",
        "Motivo",
        # 'Motivo_O', 'motivos',
        "Modo Agrupado",
        "fecha_inicio",
        "fecha_termino",
        "duracion",
        # 'Tiempo Tot de Viaje', # same as duracion, verified
        # 'TipoEstacionamiento', 'TpoBusqueda', 'TpoEstacionadoHH',
        # 'TpoEstacionadoMM', 'CostoEstacionamiento',
        "FACTOR",
    ]

    # Legs columns
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

    m_cols = m1_cols + m2_cols + m3_cols + m4_cols + m5_cols + m6_cols

    trips = od_df[trip_cols + m_cols].drop(0, level="VIAJE")

    trips = (
        od_df[trip_cols + m_cols]
        .drop(0, level="VIAJE")
        .pipe(
            lambda df: (
                df.droplevel("VIAJE").set_index(
                    (df.groupby(["HOGAR", "HABITANTE"]).cumcount() + 1).rename("VIAJE"),
                    append=True,
                )
            )
        )
    )

    legs_wide = trips[m_cols].copy()
    trips = trips.drop(columns=m_cols).copy()

    # First trips that do not begin home have unknown origin.
    # Checked by hand, all assignments make sense
    trips.loc[(slice(None), slice(None), 1), "Origen"] = (
        trips.loc[(slice(None), slice(None), 1)]
        .pipe(lambda df: df.where(df.Origen.isin(["Hogar"]), "Otro"))
        .Origen
    )

    # Change destino del viaje anterior por actual valor
    # This enables chain checks and fixes
    old_idx = trips.loc[trips.Origen == "el destino de viaje inmediato anterior"].index
    new_idx = pd.MultiIndex.from_arrays(
        [
            old_idx.get_level_values(0),
            old_idx.get_level_values(1),
            old_idx.get_level_values(2) - 1,
        ]
    )
    trips.loc[old_idx, "Origen"] = trips.loc[new_idx, "Destino"].values

    # Fix Nans, mostly by hand
    trips.loc[[("22899-4", 2, 2), ("36217-2", 1, 2)], "Destino"] = "Hogar"
    trips.loc[("4819-703", 3, 1), ["Destino", "Motivo"]] = [
        "Tienda/(Super)mercado",
        "compras",
    ]
    trips.loc[("4819-703", 3, 2), "Destino"] = "Hogar"
    trips.loc[(trips.Motivo.isna()) & (trips.Destino == "Otro"), "Motivo"] = "otro"
    trips.loc[
        (trips.Motivo.isna()) & (trips.Destino == "Lugar de Trabajo"), "Motivo"
    ] = "trabajo"
    trips.loc[
        (trips.Motivo.isna()) & (trips.Destino == "Tienda/(Super)mercado"), "Motivo"
    ] = "compras"

    # Fix chains in Origen Destino
    # Also mostly by hand
    fix_od_chains(trips)
    # It is not posible to fix all
    # Some inhabitants are missing trips
    missing_trips = check_od_chains(trips)

    # Now that first trip origin is correct, and
    # trips are chained. We look at forbbiden transitions
    # in origin->destination and in trip purpose.

    # Trips from Home to Home
    home_to_home = trips.query("Destino == 'Hogar' & Origen == 'Hogar'")
    assert len([i for i in home_to_home.index.droplevel(2) if i in missing_trips]) == 0

    # If motivo == trabajo, change destino -> Lugar de Trabajo
    # and origin of next trip to lugar de trabajo
    home_to_home_tr = home_to_home.query("Motivo == 'trabajo'")
    trips.loc[home_to_home_tr.index, "Destino"] = "Lugar de Trabajo"
    trips.loc[index_next_trip(home_to_home_tr.index), "Origen"] = "Lugar de Trabajo"

    home_to_home = trips.query("Destino == 'Hogar' & Origen == 'Hogar'")
    assert np.all(check_od_chains(trips) == missing_trips)

    # For motivo acompañar / recoger, by hand fix
    trips.loc[
        ("42779-12", 2, slice(9, 10)), ["ZonaOri", "ZonaDest", "Origen", "Destino"]
    ] = [[231, 232, "Hogar", "Otro"], [232, 231, "Otro", "Hogar"]]
    home_to_home = trips.query("Destino == 'Hogar' & Origen == 'Hogar'")
    assert np.all(check_od_chains(trips) == missing_trips)

    # Other home-home trips seem to be walks or car short trips
    # that may be fine
    # Or dates seem weird, leave as is, but keep track of them
    home_to_home = trips.query("Destino == 'Hogar' & Origen == 'Hogar'")

    # Last trip number
    trips["last"] = (
        trips.reset_index()
        .groupby(["HOGAR", "HABITANTE"])
        .VIAJE.transform("max")
        .values
    )

    # Trips going home with a purpose not return to home
    to_home_other_purpose = trips.query(
        "Destino == 'Hogar' & Motivo != 'regreso a casa'"
    )

    # Trusting the chain of Origen Destino, we change purpose.
    # Most cases asre last trip anyway.
    # A few of them are not to Home TAZ, but home taz seems wrong,
    # need to check this further down
    # For now, change all
    (to_home_other_purpose.TAZ != to_home_other_purpose.ZonaDest).pipe(
        lambda s: s[s].index
    )

    trips.loc[to_home_other_purpose.index, "Motivo"] = "regreso a casa"

    assert np.all(check_od_chains(trips) == missing_trips)

    # Trips with purpose return to home not going home
    to_home_not_home = trips.query("Destino != 'Hogar' & Motivo == 'regreso a casa'")

    # None of these is last trip
    # We can try to impute purpose based on destination
    # Destination Otro or Otro hogar, impute otro
    trips.loc[
        to_home_not_home.query("Destino == 'Otro' | Destino == 'Otro hogar'").index,
        "Motivo",
    ] = "otro"
    to_home_not_home = trips.query("Destino != 'Hogar' & Motivo == 'regreso a casa'")

    # People going to hospitals, checked manually
    trips.loc[
        to_home_not_home.query("Destino == 'Farmacia/Clínica/Hospital'").index, "Motivo"
    ] = "salud"
    to_home_not_home = trips.query("Destino != 'Hogar' & Motivo == 'regreso a casa'")

    # Students got to school to study, housewifes acompany
    trips.loc[
        to_home_not_home.query(
            "Destino == 'Escuela' & Ocupacion == 'Estudiante'"
        ).index,
        "Motivo",
    ] = "estudios"
    trips.loc[
        to_home_not_home.query(
            "Destino == 'Escuela' & Ocupacion == 'Ama de casa'"
        ).index,
        "Motivo",
    ] = "acompañar / recoger"
    trips.loc[
        to_home_not_home.query("Destino == 'Escuela' & Ocupacion == 'Otro'").index,
        "Motivo",
    ] = "otro"
    to_home_not_home = trips.query("Destino != 'Hogar' & Motivo == 'regreso a casa'")

    # Destino lugar de trabajo is work trip as per LugarDest
    trips.loc[
        to_home_not_home.query("Destino == 'Lugar de Trabajo'").index, "Motivo"
    ] = "trabajo"
    to_home_not_home = trips.query("Destino != 'Hogar' & Motivo == 'regreso a casa'")

    # Destino tienda is shopping trip as per LugarDest
    trips.loc[
        to_home_not_home.query("Destino == 'Tienda/(Super)mercado'").index, "Motivo"
    ] = "compras"
    to_home_not_home = trips.query("Destino != 'Hogar' & Motivo == 'regreso a casa'")

    # TODO. The following are a little more complicated
    # A trip to school that's not to study can be valid if
    # 1. Adult goes to work
    # 2. An adult acompanies a kid
    # 3. Adult goes to other such as a parent meeting.
    trips.query(
        "Destino == 'Escuela' "
        "& Motivo != 'estudios' "
        "& Motivo != 'acompañar / recoger'"
    ).Motivo.value_counts()

    # TODO. Study trips that do not go to a School alse warrant investigation
    trips.query("Destino != 'Escuela' & Motivo == 'estudios'").Destino.value_counts()

    # TODO. Health trips not to a hospital or other
    trips.query("Motivo == 'salud'").Destino.value_counts()

    # Now we look at conflicting purpose chains
    # Lets get the pupose transition matrix, ignoring people with missing trips
    get_purpose_tmat(trips, ignore=missing_trips)
    # The regreso a hogar chained trips are the Hogar->Hogar trips,
    # which can happen OK

    # TODO. How many small kids move alone?
    # Is there an adult that travels to the same location at the same time?
    # If so, should we assume Motivo == 'acompañar'?

    # Look at conflicting home locations, there seems to be some typos
    # Lat lon coordinates usually point at TAZ
    # Should we trust TAZ and change ZonaOri and ZonaDest for Home?
    print(
        "Conflicting home location ",
        trips.query(
            "(Destino == 'Hogar' & TAZ != ZonaDest) "
            "| (Origen == 'Hogar' & TAZ != ZonaOri)"
        ).shape,
    )

    habs = trips.groupby(["HOGAR", "HABITANTE"]).first().copy()

    habs["hogares"] = (
        trips.query("Destino == 'Hogar'")
        .groupby(["HOGAR", "HABITANTE"])
        .ZonaDest.unique()
        .reindex(habs.index)
        .apply(lambda l: l.tolist() if isinstance(l, np.ndarray) else [])
        + trips.query("Origen == 'Hogar'")
        .groupby(["HOGAR", "HABITANTE"])
        .ZonaOri.unique()
        .reindex(habs.index)
        .apply(lambda l: l.tolist() if isinstance(l, np.ndarray) else [])
    ).apply(np.unique)

    habs["n_hogares"] = habs.hogares.apply(len)

    habs["TAZ_in_hogares"] = [d in l for d, l in zip(habs.TAZ, habs.hogares)]

    print("n_hogares > 1", habs.query("n_hogares > 1").shape)
    print("not TAZ_in_hogares", habs.query("not TAZ_in_hogares").shape)
    print(
        "n_hogares > 1 & not TAZ_in_hogares",
        habs.query("n_hogares > 1 & not TAZ_in_hogares").shape,
    )
    habs_problems = habs.query("n_hogares > 1 | not TAZ_in_hogares").copy()
    assert (
        len([i for i in habs_problems.index.get_level_values(0) if i in missing_trips])
        == 0
    )

    for hogar, habitante in habs_problems.index:
        fix_home_loc(trips, hogar, habitante)

    trips["Modo Agrupado"] = trips["Modo Agrupado"].str.strip().str.lower()

    trips.loc[
        (trips.Motivo == "estudios")
        & (trips["Modo Agrupado"] == "modos combinados sin tpub"),
        "Modo Agrupado",
    ] = [
        "transporte escolar",
        "transporte escolar",
        "uber, cabify , didi o similar",
        "transporte escolar",
        "transporte escolar",
        "transporte escolar",
        "transporte escolar",
        "tpub",
        "tpub",
        "tpub",
    ]

    trips_next_idx = trips.index.intersection(index_next_trip(trips.index))
    trips_prev_idx = index_prev_trip(trips_next_idx)
    trips["stay_duration_h"] = (
        trips.loc[trips_next_idx].fecha_inicio.values
        - trips.loc[trips_prev_idx].fecha_termino
    ).dt.total_seconds() / 3600

    trips = trips.drop(columns="Ocupacion")

    trips["ntrip"] = np.nan
    trips.loc[trips.Motivo == "estudios", "ntrip"] = (
        trips.loc[trips.Motivo == "estudios"]
        .groupby(["HOGAR", "HABITANTE"])
        .Motivo.transform("rank", method="first")
    )
    trips.loc[trips.Motivo == "trabajo", "ntrip"] = (
        trips.loc[trips.Motivo == "trabajo"]
        .groupby(["HOGAR", "HABITANTE"])
        .Motivo.transform("rank", method="first")
    )
    print(
        "Conflicting home location after fixes",
        trips.query(
            "(Destino == 'Hogar' & TAZ != ZonaDest) "
            "| (Origen == 'Hogar' & TAZ != ZonaOri)"
        ).shape,
    )

    return trips, legs_wide
