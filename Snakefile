rule get_mg:
    output:
        "data/19_nuevoleon.gpkg"
    shell:
        "curl -o data/19_nuevoleon.gpkg "
        "https://microsimulationtasha.blob.core.windows.net/"
        "population-synthesis/19_nuevoleon.gpkg"

rule generate_assignments:
    input:
        rules.get_mg.output
    output:
        "data/outputs/taz_assignment.yaml",
        "data/outputs/taz_assign_report.pdf"
    run:
        from od_mty_2019 import generate_taz_assignment
        import matplotlib
        matplotlib.use('PDF')
        generate_taz_assignment()

rule train_model:
    output:
        "data/outputs/informal_model.pkl"
    run:
        from od_mty_2019.informal_model import train_model
        train_model()

rule od_clean:
    input:
        rules.train_model.output
    output:
        "data/outputs/od_clean/trips.csv",
        "data/outputs/od_clean/people.csv",
        "data/outputs/od_clean/households.csv"
    run:
        from od_mty_2019 import generate_od_tables
        generate_od_tables()