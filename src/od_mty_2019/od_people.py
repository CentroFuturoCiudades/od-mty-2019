"""Generate the people table from a clean OD dataframe."""

from importlib import resources

import numpy as np
import pandas as pd
import yaml

with resources.path("od_mty_2019", "od_map_parentesco.yaml") as p:
    with open(p, "r", encoding="utf-8") as f:
        parentesco_map = yaml.load(f, yaml.SafeLoader)
with resources.path("od_mty_2019", "od_map_dis.yaml") as p:
    with open(p, "r", encoding="utf-8") as f:
        dis_map = yaml.load(f, yaml.SafeLoader)
from .sector_maps import ocu_only_map, sect_map


def get_educ_asi(r):
    """Infer current school attending level from maximum previously
    attained level and age."""

    educ = r.EDUC
    edad = r.Edad
    if r.ASISTEN == 0:
        return "Blanco por pase"
    if educ == "Sin Educación":
        return "Básica"
    if educ == "Básica":
        if edad <= 14:
            return "Básica"
        return "MediaSup"
    if educ == "MediaSup":
        if edad < 18:
            return "MediaSup"
        return "Superior"
    if educ == "Superior":
        return "Superior"
    raise NotImplementedError


def build_people_table(od_df, trips, add_informal=False):
    """Builds the people table from the od survey. Cleans up many problems."""

    people_cols = [
        "MUN",
        "TAZ",
        "Género",
        "Edad",
        "RelaciónHogar",
        "Discapacidad",
        "Estudios",
        "Estudios_O",
        "Ocupacion",
        "Ocupacion_O",
        "SectorEconom",
        "SectorEconom_O",
        # 'FACTOR',
    ]

    # Agregate the people table, trust first variables
    people = od_df[people_cols].groupby(["HOGAR", "HABITANTE"]).first().copy()
    people["Ocupacion"] = people.Ocupacion.str.strip().str.lower()

    # Auxiliary dataframe to find conflicting people values
    conf_people = (
        od_df.reset_index(level=2)[people_cols]
        .groupby(["HOGAR", "HABITANTE"])
        .nunique()
    )

    # Género -> SEXO
    people["SEXO"] = people.Género.map({"Mujer": "F", "Hombre": "M"})
    people = people.drop(columns="Género")

    # Edad -> EDAD
    people["EDAD"] = pd.cut(
        people.Edad,
        (0, 3, 5, 6, 8, 12, 15, 18, 25, 50, 60, 65, 131),
        right=False,
        labels=[
            "0-2",
            "3-4",
            "5",
            "6-7",
            "8-11",
            "12-14",
            "15-17",
            "18-24",
            "25-49",
            "50-59",
            "60-64",
            "65-130",
        ],
    )

    # RelacionHogar -> PARENTESCO
    # This variable is not required by the model,
    # but matches CENSUS, used for imputation

    # Fix duplicated and missing Jefe de Familia in RELACION HOGAR
    is_head_idx = (
        od_df.reset_index(level=2)
        .loc[(conf_people > 1).T.sum() > 0, people_cols]
        .query('RelaciónHogar == "Jefe(a) de familia"')
        .index
    )
    people.loc[is_head_idx, "RelaciónHogar"] = "Jefe(a) de familia"

    # Find vive solo, if first inhabitante change to jefe,
    # else change to Sin parentesco
    people.loc[
        (
            (people.RelaciónHogar == "Vive solo(a) / Independiente")
            & (people.index.get_level_values(1) == 1)
        ),
        "RelaciónHogar",
    ] = "Jefe(a) de familia"

    people.loc[
        (
            (people.RelaciónHogar == "Vive solo(a) / Independiente")
            & (people.index.get_level_values(1) > 1)
        ),
        "RelaciónHogar",
    ] = "Otro"

    # Remap
    for k, v in parentesco_map.items():
        people["RelaciónHogar"] = people.RelaciónHogar.replace(v, k)
    people["PARENTESCO"] = people.RelaciónHogar.replace("No especificado", np.nan)
    people = people.drop(columns="RelaciónHogar")

    # Note, Jefe can be missing in OD, since households are not complete.
    # Found 4 missing jefes
    # Just one Jefe is recovered

    # Discapacidad -> DIS
    people["DIS"] = people.Discapacidad.map(dis_map)
    people = people.drop(columns="Discapacidad")

    # Estudios -> EDUC
    people.loc[people.Estudios == "Otro", "Estudios"] = people.loc[
        people.Estudios == "Otro", "Estudios_O"
    ]
    people["EDUC"] = (
        people.Estudios.replace(
            ["Sin instrucción", "Preescolar", "Sin Educación"], "Sin Educación"
        )
        .replace("Primaria o Secundaria", "Básica")
        .replace(["Carrera técnica o preparatoria", "Preparatoria Trunca"], "MediaSup")
        .replace(
            [
                "Licenciatura",
                "Postgrado",
                "Carrera de Química Trunca",
                "4 Semestres facultad",
            ],
            "Superior",
        )
        .replace(["Educación Especial", "Esc especial"], "Sin Educación")
        .replace(["No dio información", "N/P", None], np.nan)
    )
    people["EDUC2"] = (
        people.Estudios.replace(
            ["Sin instrucción", "Preescolar", "Sin Educación"], "Sin Educación"
        )
        .replace("Primaria o Secundaria", "Básica")
        .replace(["Carrera técnica o preparatoria", "Preparatoria Trunca"], "MediaSup")
        .replace(
            ["Licenciatura", "Carrera de Química Trunca", "4 Semestres facultad"],
            "Licenciatura",
        )
        .replace(["Educación Especial", "Esc especial"], "Sin Educación")
        .replace(["No dio información", "N/P", None], np.nan)
    )

    # ASISTEN
    # People who have student as ocupation or that realize a study trip
    people["ASISTEN"] = 0
    idx_study_trips = trips.query("Motivo == 'estudios'").index.droplevel(2).unique()
    people.loc[idx_study_trips, "ASISTEN"] = 1
    people.loc[people.Ocupacion == "estudiante", "ASISTEN"] = 1
    # NOTE: There are still several inconsistencies
    # regarding trips start and end times and stay duration.
    # This make identifying part time students hard.
    # We can try to using stay_duration_h but this information
    # is not in the census.
    # As an alternative we can look at the ocupations status.
    # Workers that study can be classified into part time students depending
    # on the working hours.
    # Need to look at the distribution of those varibales in census.

    # TAZ_ASI
    # TAZ where they attend school
    # 15 inhabitants have two different destination zones for study trip.
    # Choose first arbitrarly
    idx_study_trips = trips.query("Motivo == 'estudios'").index
    taz_asi = (
        trips.loc[idx_study_trips, "ZonaDest"].groupby(["HOGAR", "HABITANTE"]).first()
    )
    people["TAZ_ASI"] = "Blanco por pase"
    people.loc[people.ASISTEN == 1, "TAZ_ASI"] = np.nan
    people.loc[taz_asi.index, "TAZ_ASI"] = taz_asi

    # EDUC_ASI
    # EDUC reports maximum completed education level.
    # Current atending level must be estimated.
    people["EDUC_ASI"] = "Blanco por pase"
    people["EDUC_ASI"] = people.apply(get_educ_asi, axis=1)

    # TIE_TRASLADO_ESCU
    people["TIE_TRASLADO_ESCU"] = "Blanco por pase"
    people.loc[people.ASISTEN == 1, "TIE_TRASLADO_ESCU"] = np.nan
    idx_study_trips = trips.query("Motivo == 'estudios'").index
    duracion_cat = pd.cut(
        (
            (trips.loc[idx_study_trips, "duracion"].dt.total_seconds() / 60)
            .groupby(["HOGAR", "HABITANTE"])
            .max()
        ),
        [-1, 15, 30, 60, 120, 1e6],
        labels=[
            "Hasta 15 minutos",
            "16 a 30 minutos",
            "31 minutos a 1 hora",
            "Más de 1 hora y hasta 2 horas",
            "Más de 2 horas",
        ],
        right=True,
    )
    people.loc[duracion_cat.index, "TIE_TRASLADO_ESCU"] = duracion_cat

    people = people.drop(columns=["Estudios", "Estudios_O"])

    # MED_TRASLADO_ESC TODO
    # This columns is realy not usefull.
    # We should keep all the used modes as in the census
    # First need to finish preprocessing the legs table.
    # med_traslado_esc = (
    #    trips.query("Motivo == 'estudios'")["Modo Agrupado"]
    #    .replace(
    #        ["automóvil (pasajero)", "automóvil (conductor)"], "Automóvil o camioneta"
    #    )
    #    .replace("bicicleta", "Bicicleta")
    #    .replace("a pie (caminando)", "Caminando")
    #    .replace("motocicleta", "Motocicleta o motoneta")
    #    .replace("uber, cabify , didi o similar", "Taxi (App Internet)")
    #    .replace("taxi", "Taxi (sitio, calle, otro)")
    #    .replace("transporte escolar", "Transporte escolar")
    #    .replace(["tpub", "modos combinados con tpub"], "TPUB")
    #    .replace(["otro", "transporte de personal"], "Otro")
    # )

    people["SectorEconom"] = (
        people.SectorEconom.str.lower()
        .str.strip()
        .replace("servicio", "servicios")
        .replace("transporte y comunicación", "transporte y comunicaciones")
    )
    people["SectorEconom_O"] = people.SectorEconom_O.str.lower().str.strip()
    people["Ocupacion_O"] = people.Ocupacion_O.str.lower().str.strip()
    people["Ocupacion"] = (
        people.Ocupacion.str.lower()
        .str.strip()
        .replace("sin instrucción", "sin empleo")
        .replace("profesionista independiente", "trabajador(a) por cuenta propia")
    )

    people.loc[
        ((people.Ocupacion == "ama de casa") & (people.SectorEconom == "otro")),
        "SectorEconom_O",
    ] = None
    people.loc[people.Ocupacion_O == "no aplica", "Ocupacion_O"] = None

    people.loc[
        (people.Ocupacion != "otro") & people.Ocupacion_O.notnull(),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = [
        ["empleado (a)", None, "servicios", None],
        ["jubilado", None, "otro", None],
        ["empleado (a)", None, "servicios", None],
        ["profesionista empleado", None, "servicios", None],
    ]

    people.loc[
        ((people.SectorEconom != "otro") & people.SectorEconom_O.notnull()),
        "SectorEconom_O",
    ] = None

    people.loc[
        ((people.Ocupacion_O == "empleado (a)")), ["Ocupacion", "Ocupacion_O"]
    ] = ["empleado (a)", None]

    people.loc[
        people.Ocupacion_O.astype(str).str.contains("taxi"),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["trabajador(a) por cuenta propia", None, "transporte y comunicaciones", None]

    people.loc[
        people.Ocupacion_O.astype(str).str.contains("uber"),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["trabajador(a) por cuenta propia", None, "transporte y comunicaciones", None]

    people.loc[
        (
            (people.Ocupacion == "otro")
            & (people.SectorEconom == "transporte y comunicaciones")
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleado (a)", None, "transporte y comunicaciones", None]

    people.loc[
        people.Ocupacion_O.isin(["campesino", "ejidatario"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["trabajador(a) por cuenta propia", None, "agricultura y ganadería", None]

    people.loc[
        ((people.Ocupacion == "otro") & (people.SectorEconom == "gobierno")),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleado (a)", None, "gobierno", None]

    people.loc[people.Ocupacion_O == "negocio propio", ["Ocupacion", "Ocupacion_O"]] = [
        "patrón(a) o empleador(a)",
        None,
    ]

    people.loc[
        (people.Ocupacion == "patrón(a) o empleador(a)")
        & (people.SectorEconom == "otro"),
        ["SectorEconom", "SectorEconom_O"],
    ] = [None, None]

    people.loc[
        (people.Ocupacion_O == "contratista") & (people.SectorEconom == "otro"),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["patrón(a) o empleador(a)", None, "construcción", None]

    people.loc[(people.Ocupacion_O == "contratista"), ["Ocupacion", "Ocupacion_O"]] = [
        "patrón(a) o empleador(a)",
        None,
    ]

    people.loc[
        ((people.Ocupacion == "otro") & (people.SectorEconom == "construcción")),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "construcción", None]

    people.loc[
        people.Ocupacion_O.isin(
            [
                "vendedor ambulante",
                "por su cuenta",
                "puesto",
                "trabaja por su cuenta",
                "empleado independiente",
                "jefe propio",
                "trabajador independiente",
                "oficio propio",
                "por cuenta propia",
            ]
        ),
        ["Ocupacion", "Ocupacion_O"],
    ] = ["trabajador(a) por cuenta propia", None]

    people.loc[
        (
            people.Ocupacion_O.isin(
                [
                    "estilista",
                    "estetica",
                    "negocio propio estética",
                    "trabaja independiente en casa estética",
                ]
            )
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["trabajador(a) por cuenta propia", None, "servicios", None]

    people.loc[
        people.Ocupacion_O.isin(["mesero (a)", "cocinero", "chef", "mecanico"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "servicios", None]

    people.loc[
        people.Ocupacion_O.isin(["tablajero", "carpintero", "construcción"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "construcción", None]

    people.loc[
        ((people.Ocupacion == "otro") & (people.SectorEconom == "comercio")),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "comercio", None]

    people.loc[
        people.Ocupacion_O.isin(["panadero"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "comercio", None]

    people.loc[
        (
            (people.Ocupacion == "otro")
            & (people.SectorEconom == "industria manufacturera")
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "industria manufacturera", None]

    people.loc[
        people.Ocupacion_O.isin(
            [
                "albañil",
                "soldador",
                "electricista",
                "eléctrico",
                "pintor",
                "herrero",
                "reparador de canceles",
                "pulidor de.pisos",
                "plomero",
                "instalador de fibra",
            ]
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "construcción", None]

    people.loc[
        people.Ocupacion_O.astype(str).str.contains("empres"),
        ["Ocupacion", "Ocupacion_O"],
    ] = ["patrón(a) o empleador(a)", None]

    people.loc[
        people.Ocupacion_O.isin(
            [
                "operador de ruta urbana",
                "transportista-fletes",
                "trailero",
                "chofer",
                "paquetera",
                "paqueter smart",
                "chofer de camión urbano",
                "chofer de transporte de personal",
                "chofer de tráiler",
                "transportista",
                "operador de transporte de autobus",
                "operador",
                "operadora",
                "operador 5 ruedas",
                "transporte escolar",
                "operador de transporte escolar",
                "chofer de transporte de personal",
            ]
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "transporte y comunicaciones", None]

    people.loc[
        people.Ocupacion_O.isin(
            [
                "trabaja en su propio transporte",
                "chofer de aplicación",
                "chofer de didi",
            ]
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["trabajador(a) por cuenta propia", None, "transporte y comunicaciones", None]

    people.loc[
        people.Ocupacion_O.isin(["fabrica de baños", "mantenimiento fabrica"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "industria manufacturera", None]

    people.loc[
        people.Ocupacion_O.isin(
            [
                "panaderia",
                "carnicero",
                "empacador",
                "almacenista",
                "jefe de almacen",
                "cargador",
                "paqueteria sorian",
            ]
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "comercio", None]

    people.loc[
        people.Ocupacion_O.isin(["publico", "servidor publico"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "gobierno", None]

    people.loc[
        ((people.Ocupacion == "otro") & (people.SectorEconom == "servicios")),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "servicios", None]

    people.loc[people.Ocupacion_O.isin(["jornalero"]), ["Ocupacion", "Ocupacion_O"]] = [
        "jornalera(o) o peón(a)",
        None,
    ]

    people["Ocupacion_O"] = people["Ocupacion_O"].replace(["n/p", "n/g", "n/r"], None)
    people["SectorEconom_O"] = people["SectorEconom_O"].replace(
        ["n/p", "n/g", "n/r", "ninguno"], None
    )

    people.loc[
        (
            (people.SectorEconom == "otro")
            & (people.Ocupacion == "comerciante")
            & (people.SectorEconom_O.isna())
        ),
        "SectorEconom",
    ] = "comercio"

    people.loc[people.Ocupacion_O == "pensionado", ["Ocupacion", "Ocupacion_O"]] = [
        "Es pensionada(o) o jubilada(o)",
        None,
    ]

    people.loc[
        people.Ocupacion_O == "mecánico",
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "servicios", None]

    people.loc[
        people.Ocupacion_O == "pensionado (a)",
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["Es pensionada(o) o jubilada(o)", None, "otro", None]

    people.loc[
        people.Ocupacion_O.isin(
            [
                "maestro",
                "maestra",
                "profesora",
                "profesor",
                "educación",
                "maestra particular",
                "asistente educativo",
                "educadora",
            ]
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "servicios", None]

    people.loc[
        people.Ocupacion_O.isin(
            [
                "trilero",
                "operador de tráiler",
                "operador de autobús",
                "operador de camión",
            ]
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "transporte y comunicaciones", None]

    people.loc[
        people.Ocupacion_O.isin(
            [
                "empleada doméstica",
                "trabajo domestico",
                "limpieza en casas",
                "trabajadora doméstica",
                "empleada domestica",
            ]
        ),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "servicios", None]

    people.loc[people.SectorEconom_O.isin(["ayudante general"]), "SectorEconom_O"] = (
        None
    )

    people.loc[
        people.Ocupacion_O.isin(["ayudante general"]), ["Ocupacion", "Ocupacion_O"]
    ] = ["empleada(o) u obrera(o)", None]

    people.loc[
        people.SectorEconom_O.isin(["pensionado"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["jubilado", None, "otro", None]

    people.loc[
        people.SectorEconom_O.isin(
            [
                "pemex",
                "refineria",
                "refinería de pemex",
                "compañía dentro de pemex",
                "refinería pemex",
                "compañías interior de refinería",
                "guardia refineria",
            ]
        ),
        ["Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = [None, "industria manufacturera", None]

    people.loc[
        ((people.Ocupacion_O == "no estudia ni trabaja")),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["sin empleo", None, "otro", None]

    people.loc[
        ((people.Ocupacion_O == "no estudia")),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["sin empleo", None, "otro", None]

    people.loc[
        people.Ocupacion_O.isin(["guardia", "guardia de seguridad", "vigilante"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "servicios", None]

    people.loc[
        ((people.Ocupacion_O == "independiente")),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["trabajador(a) por cuenta propia", None, "otro", None]

    people.loc[
        ((people.Ocupacion_O == "jardinero")),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["trabajador(a) por cuenta propia", None, "servicios", None]

    people.loc[
        ((people.SectorEconom_O == "salud")), ["SectorEconom", "SectorEconom_O"]
    ] = ["servicios", None]

    people.loc[
        ((people.SectorEconom_O == "fabrica")), ["SectorEconom", "SectorEconom_O"]
    ] = ["industria manufacturera", None]

    people.loc[
        ((people.SectorEconom_O == "industria alimenticia")),
        ["SectorEconom", "SectorEconom_O"],
    ] = ["industria manufacturera", None]

    people.loc[
        ((people.SectorEconom_O.isin(["educación", "educacion"]))),
        ["SectorEconom", "SectorEconom_O"],
    ] = ["servicios", None]

    people.loc[
        people.Ocupacion_O.isin(["policía municipal"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "gobierno", None]

    people.loc[
        people.SectorEconom_O.isin(["recolector de metal"]),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = ["empleada(o) u obrera(o)", None, "comercio", None]

    people.loc[
        people.SectorEconom_O.isin(["albañil"]), ["SectorEconom", "SectorEconom_O"]
    ] = ["construcción", None]

    people.loc[
        people.SectorEconom_O.isin(["hacen chocolates"]),
        ["SectorEconom", "SectorEconom_O"],
    ] = ["industria manufacturera", None]

    people.loc[people.SectorEconom_O.isin(["privado"]), "SectorEconom_O"] = None

    people.loc[
        people.Ocupacion_O.isin(
            [
                "estudiante universitario",
                "estudia y trabaja",
                "trabaja y estudia",
                "estudiante y trabaja",
            ]
        ),
        "ASISTEN",
    ] = 1

    for r, o_list in ocu_only_map.items():
        for ocu in o_list:
            people.loc[
                (
                    (people.Ocupacion == "otro")
                    & (people.SectorEconom_O.isna())
                    & (people.SectorEconom == "otro")
                    & (people.Ocupacion_O == ocu)
                ),
                ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
            ] = r

    people.loc[
        ((people.Ocupacion_O.notnull())),
        ["Ocupacion", "Ocupacion_O", "SectorEconom", "SectorEconom_O"],
    ] = [
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["otro", None, "otro", None],
        ["empleado (a)", None, "comercio", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "industria manufacturera", None],
        ["empleado (a)", None, "comercio", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "comercio", None],
        ["empleado (a)", None, "comercio", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "industria manufacturera", None],
        ["empleado (a)", None, "comercio", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
        ["empleado (a)", None, "servicios", None],
    ]

    for secto, sect in sect_map.items():
        people.loc[
            ((people.SectorEconom_O == secto)), ["SectorEconom", "SectorEconom_O"]
        ] = [sect, None]

    people.loc[
        (people.SectorEconom_O.notnull() & (people.Ocupacion == "comerciante")),
        ["Ocupacion", "SectorEconom", "SectorEconom_O"],
    ] = ["trabajador(a) por cuenta propia", "comercio", None]

    people.loc[
        (people.SectorEconom_O.notnull()),
        ["Ocupacion", "SectorEconom", "SectorEconom_O"],
    ] = [
        ["trabajador(a) por cuenta propia", "comercio", None],
        ["trabajador(a) por cuenta propia", "construcción", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "servicios", None],
        ["trabajador(a) por cuenta propia", "servicios", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "transporte y comunicaciones", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "otro", None],
        ["trabajador(a) por cuenta propia", "comercio", None],
        ["trabajador(a) por cuenta propia", "comercio", None],
        ["trabajador(a) por cuenta propia", "construcción", None],
        ["trabajador(a) por cuenta propia", "comercio", None],
        ["trabajador(a) por cuenta propia", "servicios", None],
        ["trabajador(a) por cuenta propia", "comercio", None],
        ["trabajador(a) por cuenta propia", "construcción", None],
        ["trabajador(a) por cuenta propia", "otro", None],
    ]

    people = people.drop(columns=["Ocupacion_O", "SectorEconom_O"])

    people["Ocupacion"] = (
        people.Ocupacion.replace(
            ["empleado (a)", "obrero(a)", "profesionista empleado", "oficinista"],
            "empleada(o) u obrera(o)",
        )
        .replace("jubilado", "Es pensionada(o) o jubilada(o)")
        .replace("jornalera(o) o peón(a)", "empleada(o) u obrera(o)")
        .replace("ama de casa", "Se dedica a los quehaceres del hogar")
        .replace("sin empleo", "No trabaja")
        .replace("comerciante", "trabajador(a) por cuenta propia")
    )
    people["SITTRA"] = people.Ocupacion.copy()
    people["CONACT"] = people.Ocupacion.copy()
    people = people.drop(columns="Ocupacion")

    people["SITTRA"] = (
        people.SITTRA.replace("Se dedica a los quehaceres del hogar", "Blanco por pase")
        .replace("Es pensionada(o) o jubilada(o)", "Blanco por pase")
        .replace("estudiante", "Blanco por pase")
        .replace("No trabaja", "Blanco por pase")
        .replace("otro", None)
    )

    people["CONACT"] = (
        people.CONACT.replace("empleada(o) u obrera(o)", "Trabajó")
        .replace("otro", "Trabajó")
        .replace("estudiante", "No trabaja")
        .replace("trabajador(a) por cuenta propia", "Trabajó")
        .replace("patrón(a) o empleador(a)", "Trabajó")
    )

    people.loc[(people.CONACT != "Trabajó"), "SectorEconom"] = "Blanco por pase"

    people = people.rename(columns={"SectorEconom": "ACTIVIDADES_C"})

    # TAZ_TRAB
    # Choose first arbitrarly
    idx_work_trips = trips.query("Motivo == 'trabajo'").index
    taz_trab = (
        trips.loc[idx_work_trips, "ZonaDest"].groupby(["HOGAR", "HABITANTE"]).first()
    )
    people["TAZ_TRAB"] = "Blanco por pase"
    people.loc[people.CONACT == "Trabajó", "TAZ_TRAB"] = np.nan
    people.loc[taz_trab.index, "TAZ_TRAB"] = taz_trab

    # TIE_TRASLADO_TRAB
    people["TIE_TRASLADO_TRAB"] = "Blanco por pase"
    people.loc[people.CONACT == "Trabajó", "TIE_TRASLADO_TRAB"] = np.nan
    idx_work_trips = trips.query("Motivo == 'trabajo'").index
    duracion_cat = pd.cut(
        (trips.loc[idx_work_trips, "duracion"].dt.total_seconds() / 60)
        .groupby(["HOGAR", "HABITANTE"])
        .max(),
        [-1, 15, 30, 60, 120, 1e6],
        labels=[
            "Hasta 15 minutos",
            "16 a 30 minutos",
            "31 minutos a 1 hora",
            "Más de 1 hora y hasta 2 horas",
            "Más de 2 horas",
        ],
        right=True,
    )
    people.loc[duracion_cat.index, "TIE_TRASLADO_TRAB"] = duracion_cat

    # Drop 4 households without family head
    # people = people.drop(['22899-4', '25131-4', '3275-118', '59886-12'])

    people["ASISTEN"] = people.ASISTEN.replace([0.0, 1.0], ["No", "Sí"])

    if add_informal:
        pass
        # people = classify_job(people)

    return people
