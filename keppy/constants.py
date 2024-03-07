# Sources used:
# [1] https://iau-a3.gitlab.io/NSFA/NSFA_cbe.html#GME2009
# [2] https://ssd.jpl.nasa.gov/planets/phys_par.html
# [3] https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
# [4] https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html

G = 6.67428e-11  # [m^3 kg^-1 s^-2] [1]

M_SUN = 1_988_500e24  # [kg] [4]
R_SUN = 695_700e3  # [m] [4]
MU_SUN = G * M_SUN

M_MERCURY = 0.330103e24  # [kg] [2]
R_MERCURY = 2439.4e3  # [m] [2]
MU_MERCURY = G * M_MERCURY

M_VENUS = 4.86731e24  # [kg] [2]
R_VENUS = 6051.8e3  # [m] [2]
MU_VENUS = G * M_VENUS

M_EARTH = 5.97217e24  # [kg] [2]
R_EARTH = 6371.0084e3  # [m] [2]
MU_EARTH = G * M_EARTH

M_MOON = 0.07346e24  # [kg] [3]
R_MOON = 1738.1e3  # [m] [3]
MU_MOON = G * M_MOON

M_MARS = 0.641691e24  # [kg] [2]
R_MARS = 3389.50e3  # [m] [2]
MU_MARS = G * M_MARS

M_JUPITER = 1898.125e24  # [kg] [2]
R_JUPITER = 69911e3  # [m] [2]
MU_JUPITER = G * M_JUPITER

M_SATURN = 568.317e24  # [kg] [2]
R_SATURN = 58232e3  # [m] [2]
MU_SATURN = G * M_SATURN

M_URANUS = 86.8099e24  # [kg] [2]
R_URANUS = 25362e3  # [m] [2]
MU_URANUS = G * M_URANUS

M_NEPTUNE = 102.4092e24  # [kg] [2]
R_NEPTUNE = 24622e3  # [m] [2]
MU_NEPTUNE = G * M_NEPTUNE
