"""Implements cleaning of the original destination survey.
This modules implements a lot of heuristics custom to the present od file.
If the od file changes, is likely heuristics will not give deriserd results.

These functions attempt to fix a series of issues in the official release of the survey,
many of which are frankly abhorrent and unexpected in a released data product.
Shame on you Transconsult.
We try to fix:
- typos
- homologation of category values
- consistency of trips sequence and transportation modes
- remove duplicate trips/households
- fix id typos
- fix taz origin destination codes in trip sequences
- fix trip origin detination purpose in trip sequences

"""

from importlib import resources

import numpy as np
import pandas as pd
import yaml

# Load mapping dicionaries as globals
# This maps replace typos and wrong variable values with the correct value
with resources.path("od_mty_2019", "od_map_dest.yaml") as p:
    with open(p, "r", encoding="utf-8") as f:
        dest_map = yaml.load(f, yaml.SafeLoader)
with resources.path("od_mty_2019", "od_map_motivos.yaml") as p:
    with open(p, "r", encoding="utf-8") as f:
        motivos_map = yaml.load(f, yaml.SafeLoader)
with resources.path("od_mty_2019", "od_map_id_typos.yaml") as p:
    with open(p, "r", encoding="utf-8") as f:
        id_typos = yaml.load(f, yaml.SafeLoader)
with resources.path("od_mty_2019", "od_map_dup_idxs.yaml") as p:
    with open(p, "r", encoding="utf-8") as f:
        dup_idxs = list(map(tuple, yaml.load(f, yaml.SafeLoader)))


def load_od(od_path):
    """Clean the od file an returns a clean DataFrame."""

    od = pd.read_csv(od_path, low_memory=False).rename(
        columns={"Cod_MunDomicilio": "MUN", "FE": "FACTOR", "Punto_zona": "TAZ"}
    )

    # Fix some houshold typos and drop duplicated trips
    od["H-P-V"] = od["H-P-V"].replace(list(id_typos.keys()), list(id_typos.values()))
    od = od[od["H-P-V"] != "Drop"]

    # Create a MultiIndex
    od[["HOGAR", "HABITANTE", "VIAJE"]] = od["H-P-V"].str.extract(
        r"(?P<HOGAR>.+)\/(?P<HABITANTE>\d{1,2})-(?P<VIAJE>\d{1,2})$"
    )
    od["HOGAR"] = od.HOGAR.str.strip().str.upper()
    od["HABITANTE"] = od["HABITANTE"].astype(int)
    od["VIAJE"] = od["VIAJE"].astype(int)
    od = od.set_index(["HOGAR", "HABITANTE", "VIAJE"]).sort_index()

    # Drop duplicates
    od = od.drop(index=dup_idxs)

    # Cleanup columns
    od["LineaTelef"] = od.LineaTelef.str.strip().replace("NO", "No").fillna("No")
    od["Internet"] = od.Internet.str.strip().replace("NO", "No").fillna("No")
    od["Género"] = od.Género.str.strip().replace(
        ["hombre", "HOMBRE", "mujer", "MUJER"], ["Hombre", "Hombre", "Mujer", "Mujer"]
    )
    od["RelaciónHogar"] = od.RelaciónHogar.str.strip().replace(
        ["Jefe (a) de Familia", "otro", "Madre/esposa", "Padre/esposo"],
        ["Jefe(a) de familia", "Otro", "Madre/Esposa", "Padre/Esposo"],
    )
    od["RelaciónHogar"] = od.RelaciónHogar.mask(
        od.RelaciónHogar == "Otro", od.RelaciónHogar_O.str.strip()
    ).fillna("Otro")
    od = od.drop(columns="RelaciónHogar_O")
    od["RelaciónHogar"] = od.RelaciónHogar.replace(
        [
            "AMIGO (A)",
            "N/P",
            "Novia Unión Libre",
            "Suegro(a)",
            "pareja",
            "HERMANO (A)",
            "NIETO (A)",
        ],
        [
            "Amigo (a)",
            "Otro",
            "Novia",
            "Suegro (a)",
            "Pareja",
            "Hermano (a)",
            "Nieto (a)",
        ],
    )
    od["Discapacidad"] = od.Discapacidad.str.strip().replace(
        ["No aplica", "Del oído", "Inmovilidad de alguna parte"],
        ["Ninguna", "Del Oído", "Inmovilidad en alguna parte del cuerpo"],
    )
    od["Estudios"] = od.Estudios.str.strip().replace(
        ["Primaria o secundaria", "Sin Instrucción"],
        ["Primaria o Secundaria", "Sin instrucción"],
    )

    od["Motivo"] = od.Motivo.str.lower()
    od["Motivo"] = od.Motivo.replace("acompañar/ recoger", "acompañar / recoger")

    # Fix different survey dates for the same household
    # this causes trips that begin end at different dates
    # and negative stay durations
    od["FechaHoraEnc"] = od.groupby(["HOGAR"]).FechaHoraEnc.transform("first")

    od["Hora Inicio V"] = pd.to_timedelta(od["Hora Inicio V"].apply(fix_time))
    od["Hora Término Viaje"] = pd.to_timedelta(od["Hora Término Viaje"].apply(fix_time))
    od["Tiempo Tot de Viaje"] = pd.to_timedelta(
        od["Tiempo Tot de Viaje"].apply(fix_time)
    )
    # Only 3 mismatches in the new OD,
    # for them to match
    od["Hora Término Viaje"] = od["Hora Inicio V"] + od["Tiempo Tot de Viaje"]

    od["duracion"] = od["Hora Término Viaje"] - od["Hora Inicio V"]

    od["FechaHoraEnc"] = pd.to_datetime(od.FechaHoraEnc.apply(fix_date))
    od["fecha_inicio"] = od["FechaHoraEnc"] + od["Hora Inicio V"]
    od["fecha_termino"] = od["fecha_inicio"] + od["duracion"]

    # Fix wrong captured taz
    od.loc["2179-11", "TAZ"] = 416
    od.loc["2180-S/N", "TAZ"] = 416
    od.loc["2195-927", "TAZ"] = 404
    od.loc["2457-231", "TAZ"] = 573
    od.loc["2601-121", "TAZ"] = 564
    od.loc["4029-137", "TAZ"] = 978

    od["Lugar_Or"] = od.Lugar_Or.str.normalize("NFKD").str.lower().str.strip()
    od["LugarDest"] = od.LugarDest.str.normalize("NFKD").str.lower().str.strip()

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
    ]
    od[viv_cols] = od[viv_cols].fillna(0)

    # Replace wrong classes in origin, destination, purpose
    # Reduce number classes

    od["Origen"] = (
        od.Lugar_Or.replace(dest_map["Hogar"], "Hogar")
        .replace(dest_map["Trabajo"], "Lugar de Trabajo")
        .replace(dest_map["Tienda/(Super)mercado"], "Tienda/(Super)mercado")
        .replace(dest_map["Escuela"], "Escuela")
        .replace(dest_map["Farmacia/Clínica/Hospital"], "Farmacia/Clínica/Hospital")
        .replace(dest_map["Otro hogar"], "Otro hogar")
        .replace(dest_map["Recreativo"], "Recreativo")
        .replace(dest_map["Religioso"], "Otro")
        .replace(dest_map["Banco"], "Otro")
        .replace(dest_map["Otro"], "Otro")
    )

    od["Destino"] = (
        od.LugarDest.replace(dest_map["Hogar"], "Hogar")
        .replace(dest_map["Trabajo"], "Lugar de Trabajo")
        .replace(dest_map["Tienda/(Super)mercado"], "Tienda/(Super)mercado")
        .replace(dest_map["Escuela"], "Escuela")
        .replace(dest_map["Farmacia/Clínica/Hospital"], "Farmacia/Clínica/Hospital")
        .replace(dest_map["Otro hogar"], "Otro hogar")
        .replace(dest_map["Recreativo"], "Recreativo")
        .replace(dest_map["Religioso"], "Otro")
        .replace(dest_map["Banco"], "Otro")
        .replace(dest_map["Otro"], "Otro")
    )

    od["motivos"] = (
        od.Motivo_O.dropna()
        .str.lower()
        .str.strip()
        .replace(motivos_map["Visita Enfermo"], "otro")
        .replace(motivos_map["Visita"], "otro")
        .replace(motivos_map["Panteón"], "otro")
        .replace(motivos_map["Veterinario"], "otro")
        .replace(motivos_map["Religión"], "otro")
        .replace(motivos_map["Pagos/Tramite/Banco/Cajero"], "otro")
        .replace(motivos_map["Recreación"], "recreación")
        .replace(motivos_map["Compras"], "compras")
        .replace(motivos_map["Comer"], "otro")
        .replace(motivos_map["Otro"], "otro")
        .replace(motivos_map["Diligencias"], "otro")
        .replace(motivos_map["Trabajo"], "trabajo")
        .replace(motivos_map["Cuidar personas"], "otro")
        .replace(motivos_map["Salud"], "salud")
        .replace(motivos_map["Hogar"], "regreso a casa")
        .replace(motivos_map["acompañar / recoger"], "acompañar / recoger")
        .replace(motivos_map["Taxi/Uber"], "otro")
        .replace(motivos_map["Estudio"], "estudios")
    )

    od.loc[od.Motivo == "otro", "Motivo"] = od.loc[od.Motivo == "otro", "motivos"]

    od["HabitantesObs"] = (
        od.reset_index().groupby("HOGAR").HABITANTE.transform("nunique").values
    )

    # purp of the origin for the first trip must be either
    # Home, Work, School or Other, adjust
    od.loc[(slice(None), slice(None), 1), "Origen"] = (
        od.loc[(slice(None), slice(None), 1)]
        .pipe(lambda df: df.where(df.Origen.isin(["Hogar", "Otro"]), "Hogar"))
        .Origen
    )

    return od


def fix_time(s):
    """Homogenize time stamps to common format."""
    if isinstance(s, float) and np.isnan(s):
        return s

    sl = len(s)
    if sl == 5:
        return s + ":00"
    if sl == 8:
        return s
    if sl == 19:
        return s[-8:] + pd.to_timedelta("1 day")
    if sl == 20:
        # This is low, but kept for backwards compatibility
        # with old versions of the survey
        return pd.to_datetime(s)

    print(sl)
    raise NotImplementedError


def fix_date(s):
    """Homogenize date stamp to common format."""
    if "/" in s:
        d, m, y = s.split("/")
        return f"{y}-{m}-{d} 00:00:00"

    return s
