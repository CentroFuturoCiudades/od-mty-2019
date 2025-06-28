"""
Implements functions to process, analyze and visualize traffic analysis zones data.

Includes the workflow to assign 2020 AGEB to a unique TAZ.
This is not a perfect assignment, since many AGEBS cross zones borders.
See function docstring for details.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

mun_d = {
    1: "Abasolo",
    2: "Agualeguas",
    3: "Los Aldamas",
    4: "Allende",
    5: "Anáhuac",
    6: "Apodaca",
    7: "Aramberri",
    8: "Bustamante",
    9: "Cadereyta Jiménez",
    10: "El Carmen",
    11: "Cerralvo",
    12: "Ciénega de Flores",
    13: "China",
    14: "Doctor Arroyo",
    15: "Doctor Coss",
    16: "Doctor González",
    17: "Galeana",
    18: "García",
    19: "San Pedro Garza García",
    20: "General Bravo",
    21: "General Escobedo",
    22: "General Terán",
    23: "General Treviño",
    24: "General Zaragoza",
    25: "General Zuazua",
    26: "Guadalupe",
    27: "Los Herreras",
    28: "Higueras",
    29: "Hualahuises",
    30: "Iturbide",
    31: "Juárez",
    32: "Lampazos de Naranjo",
    33: "Linares",
    34: "Marín",
    35: "Melchor Ocampo",
    36: "Mier y Noriega",
    37: "Mina",
    38: "Montemorelos",
    39: "Monterrey",
    40: "Parás",
    41: "Pesquería",
    42: "Los Ramones",
    43: "Rayones",
    44: "Sabinas Hidalgo",
    45: "Salinas Victoria",
    46: "San Nicolás de los Garza",
    47: "Hidalgo",
    48: "Santa Catarina",
    49: "Santiago",
    50: "Vallecillo",
    51: "Villaldama",
}

met_zone = [
    "Monterrey",
    "Guadalupe",
    "Apodaca",
    "San Nicolás de los Garza",
    "General Escobedo",
    "Santa Catarina",
    "Juárez",
    "García",
    "Pesquería",
    "San Pedro Garza García",
    "Cadereyta Jiménez",
    "Santiago",
    "Salinas Victoria",
    "Hidalgo",
    "General Zuazua",
    "Ciénega de Flores",
    "El Carmen",
    "Abasolo",
]


def load_marco_geo(marco_geo_path):
    """Loads AGEBS (polygon) and Localities (points) from Marco Geoestadistico
    2020 for a single state.
    Only localities that are not subdivided into AGEBs are returned, otherwise
    the composing AGEBs are returned.

    Does not verify population totals, for a function that loads population data
    see population synthesis repository.

    Returns a single GeoDataFrame indexed by Municipality, Locality and Ageb.
    """

    # Agebs geometries
    mg_agebs = gpd.read_file(marco_geo_path, layer="19a")
    mg_agebs = mg_agebs.drop(columns=["CVE_ENT"])
    mg_agebs[["CVE_MUN", "CVE_LOC"]] = mg_agebs[["CVE_MUN", "CVE_LOC"]].astype(int)
    mg_agebs = mg_agebs.rename(
        columns={"CVE_MUN": "MUN", "CVE_LOC": "LOC", "CVE_AGEB": "AGEB"}
    )
    mg_agebs["MUN"] = mg_agebs.MUN.map(mun_d)
    mg_agebs = mg_agebs.set_index(["MUN", "LOC", "AGEB"]).sort_index()

    # Localities, polygons
    mg_loc = gpd.read_file(marco_geo_path, layer="19l")
    mg_loc = mg_loc.drop(columns=["CVE_ENT", "NOMGEO", "AMBITO"])
    mg_loc[["CVE_MUN", "CVE_LOC"]] = mg_loc[["CVE_MUN", "CVE_LOC"]].astype(int)
    mg_loc = mg_loc.rename(columns={"CVE_MUN": "MUN", "CVE_LOC": "LOC"})
    mg_loc["MUN"] = mg_loc.MUN.map(mun_d)
    mg_loc = mg_loc.set_index(["MUN", "LOC"]).sort_index()

    # Localities, rural, points
    mg_loc_pr = gpd.read_file(marco_geo_path, layer="19lpr")
    mg_loc_pr = mg_loc_pr.drop(
        columns=["CVE_ENT", "NOMGEO", "PLANO", "CVE_MZA", "CVE_AGEB"]
    )
    mg_loc_pr[["CVE_MUN", "CVE_LOC"]] = mg_loc_pr[["CVE_MUN", "CVE_LOC"]].astype(int)
    mg_loc_pr = mg_loc_pr.rename(columns={"CVE_MUN": "MUN", "CVE_LOC": "LOC"})
    mg_loc_pr["MUN"] = mg_loc_pr.MUN.map(mun_d)
    mg_loc_pr = mg_loc_pr.set_index(["MUN", "LOC"]).sort_index()
    mg_loc_pr["CVEGEO"] = mg_loc_pr.CVEGEO.str[:9]

    # Point localities need to. be converted to polygons
    # as to allow for overlay operations in the whole geometries
    # This mean we are including in each taz all rural point localities within 100 m
    # of it.
    mg_loc_pr["geometry"] = mg_loc_pr.geometry.buffer(100)

    # Merge polygon and point localities, for total localities
    # Filter out localities also present is agebs
    mg_loc = mg_loc.join(mg_loc_pr, rsuffix="_pr", how="outer")
    mg_loc["CVEGEO"] = mg_loc.CVEGEO.mask(mg_loc.CVEGEO.isna(), mg_loc.CVEGEO_pr)
    mg_loc["geometry"] = mg_loc.geometry.mask(
        mg_loc.geometry.isna(), mg_loc.geometry_pr
    )
    mg_loc = mg_loc.drop(columns=["CVEGEO_pr", "geometry_pr"])
    locs_in_agebs = (
        mg_agebs.reset_index().groupby("CVEGEO").first().set_index(["MUN", "LOC"]).index
    )
    mg_loc = mg_loc.drop(locs_in_agebs)

    # Join loc and agebs data
    mg_concat = pd.concat(
        [(mg_loc.assign(AGEB="0000").set_index("AGEB", append=True)), mg_agebs]
    ).sort_index()

    # Remove duplicated localities, the ones that have been split

    return mg_concat


def merge_mg_taz(mun, taz, mg):
    """For given mun (str), assign agebs/localities in mg to a TAZ in taz.
    Each geometry in mg is assigned to the TAZ for which its overlap is largest.
    If there is no overlap, the geometry is assigned to a fictious TAZ=-10.

    Return a GeoDataFrame with the assignments.
    """
    taz_mun = taz[taz.MUNICIPIO == mun].copy().drop(columns=["CVEGEO", "ESTADO"])
    mg_mun = mg.loc[mun].copy()

    mg_mun["mg_AREA"] = mg_mun.area

    overlay = gpd.overlay(taz_mun, mg_mun.reset_index(), keep_geom_type=False).drop(
        columns=["MUNICIPIO", "ID", "AREA", "MACROZONA"]
    )

    overlay["intersection_AREA"] = overlay.area

    overlay["ratio"] = overlay.intersection_AREA / overlay.mg_AREA
    # Keep only a single mg result (CVEGEO")
    # an ageb or locality can only be assigned one taz
    overlay = overlay.sort_values("ratio", ascending=False).drop_duplicates(
        subset="CVEGEO", keep="first"
    )
    overlay = overlay.set_index(["LOC", "AGEB"]).sort_index()

    # Add unassigned mg geometries
    # Assigned all geoemtries not in a taz to taz -10
    # This way, all the population of the municipality is taken into account
    mg_unass = mg_mun.drop(overlay.index)
    mg_unass["ZONA"] = -10
    mg_unass["ratio"] = 0

    overlay = pd.concat([overlay, mg_unass]).loc[mg_mun.index]

    assert np.all(overlay.index == mg_mun.index)

    overlay["geometry"] = mg_mun.geometry

    return overlay


def plot_taz_mg(mg_gdf, taz_gdf, title, ax):
    """Plots assigned geometries in the same color with the taz outlines
    in black bold.
    Unassigned geometries are not plotted.
    """

    mg_gdf = mg_gdf.query("ratio > 0")

    mg_gdf.plot(column="ZONA", edgecolor="none", categorical=False, cmap="prism", ax=ax)

    taz_gdf.plot(
        linewidth=3,
        color="none",
        edgecolor="black",
        ax=ax,
    )

    mg_gdf.plot(color="none", edgecolor="grey", ax=ax)

    ax.set_title(title)
    ax.axis("off")


def plot_taz_mg_unass(mg_gdf, taz_gdf, title, ax):
    """Plots geometries not assigned to any TAZ with taz geometries in light blue."""

    mg_gdf = mg_gdf.query("ZONA < 0")

    taz_gdf.plot(
        color="lightblue",
        edgecolor="none",
        ax=ax,
        alpha=0.5,
    )

    mg_gdf.plot(
        color="red",
        edgecolor="none",
        categorical=False,
        ax=ax,
        legend=True,
    )

    taz_gdf.plot(
        linewidth=2,
        color="none",
        edgecolor="black",
        ax=ax,
    )

    mg_gdf.plot(color="none", edgecolor="grey", ax=ax)

    ax.set_title(title)
    ax.axis("off")


def plot_taz_empty_mg(mg_gdf, taz_gdf, title, ax):
    """Plots TAZ with no assigned geometries with bold edge.
    Other TAZ are not plotted."""

    mg_gdf = mg_gdf.query("ratio > 0")

    taz_gdf = taz_gdf[~taz_gdf.ZONA.isin(mg_gdf.ZONA.unique())]

    mg_gdf.plot(column="ZONA", edgecolor="none", categorical=False, cmap="prism", ax=ax)

    taz_gdf.plot(
        linewidth=3,
        color="none",
        edgecolor="black",
        ax=ax,
    )

    mg_gdf.plot(color="none", edgecolor="grey", ax=ax)

    ax.set_title(title)
    ax.axis("off")


def plot_chull(taz_gdf, ax):
    """Identifies TAZ with multiple parts."""

    is_multi = taz_gdf.explode().groupby("ZONA").size().pipe(lambda s: s[s > 1]).index

    taz_gdf.plot(
        linewidth=1,
        color="lightblue",
        edgecolor="black",
        ax=ax,
    )

    taz_gdf[taz_gdf.ZONA.isin(is_multi)].plot(
        linewidth=1, column="ZONA", edgecolor="black", ax=ax, categorical=True
    )

    taz_gdf[taz_gdf.ZONA.isin(is_multi)].convex_hull.plot(
        linewidth=1,
        edgecolor="red",
        color="none",
        ax=ax,
    )
    ax.axis("off")


def generate_pdf_report(taz, mg, outdir):
    """Generates a pdf report of census geometry assignment to TAZ."""
    overlay_dict = {}

    taz_dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with PdfPages(outdir / "taz_assign_report.pdf") as pdf:
            for mun in met_zone:
                taz_mun = taz[taz.MUNICIPIO == mun].copy()
                taz_dict[mun] = taz_mun.set_index("ZONA").sort_index()
                overlay = merge_mg_taz(mun, taz, mg)
                overlay_dict[mun] = overlay

                _, ax = plt.subplots(2, 2, figsize=(20, 20))

                plot_taz_mg(overlay, taz_mun, title=mun, ax=ax[0, 0])
                plot_taz_mg_unass(
                    overlay, taz_mun, title="Unassigned AGEBS", ax=ax[0, 1]
                )
                plot_taz_empty_mg(overlay, taz_mun, title="Empty TAZs", ax=ax[1, 0])
                plot_chull(taz_mun, ax=ax[1, 1])

                pdf.savefig()

                plt.close()


def generate_taz_assignment():
    """Utility function to genereta assignment artifacts."""

    mg = load_marco_geo(Path("data/19_nuevoleon.gpkg"))
    taz = gpd.read_file("data/TAZ/Zonas.gpkg").to_crs(mg.crs)
    assign_dict = {}
    for mun in met_zone:
        overlay = merge_mg_taz(mun, taz.to_crs(mg.crs), mg)
        assign_dict[mun] = (
            overlay.reset_index()
            .set_index("ZONA")
            .groupby("ZONA")
            .CVEGEO.agg(list)
            .to_dict()
        )

    generate_pdf_report(taz, mg, Path("data/outputs/"))

    with open("data/outputs/taz_assignment.yaml", "w", encoding="utf-8") as f:
        yaml.dump(assign_dict, f, allow_unicode=True)
    print("4")
