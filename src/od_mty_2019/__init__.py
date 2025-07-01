"""Scripts to analyize the origin desitnation survey for Monterrey 2019."""

from pathlib import Path

from .od_clean import load_od
from .od_households import build_household_table
from .od_people import build_people_table
from .od_trips import build_trips
from .taz import generate_taz_assignment


def generate_od_tables():
    """Generate clean OD tables."""

    opath = Path("data/outputs/od_clean/")
    opath.mkdir(exist_ok=True)

    od_df = load_od("data/PIMUS/pimus_final.csv")

    trips, _ = build_trips(od_df)
    people = build_people_table(od_df, trips, add_informal=True)
    households = build_household_table(od_df, people)

    trips.to_csv(opath / "trips.csv")
    people.to_csv(opath / "people.csv")
    households.to_csv(opath / "households.csv")
