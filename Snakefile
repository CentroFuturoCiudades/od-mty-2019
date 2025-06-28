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

rule get_mg:
    output:
        "data/19_nuevoleon.gpkg"
    shell:
        "curl -o data/19_nuevoleon.gpkg "
        "https://microsimulationtasha.blob.core.windows.net/"
        "population-synthesis/19_nuevoleon.gpkg"