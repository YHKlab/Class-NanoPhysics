"""
semi1d: modules of the one-dimensional semiconductor simulator used in Labs 4-6.

Each module holds the equation of one textbook chapter (Neamen, 4th ed.) with its numerical solution and the
closed-form results of that chapter. The labs add modules as the course proceeds:

    module             chapter     Lab 4 (Ch. 5)   Lab 5 (Ch. 6)   Lab 6 (Ch. 7-8)
    statistics         Ch. 3-4         x               x               x
    drift_diffusion    Ch. 5           x               x               x
    continuity         Ch. 6           -               x               x
    poisson            Ch. 7           -               -               x

Materials, devices, the coupling of the modules, analysis, plots and the launcher are written in each lab
notebook. A notebook adds the folder above it to sys.path and imports the modules it needs, e.g.
    from semi1d import statistics, drift_diffusion
"""
