# Sources used:
# [1] https://iau-a3.gitlab.io/NSFA/NSFA_cbe.html#GME2009
# [2] https://ssd.jpl.nasa.gov/planets/phys_par.html
# [3] https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
# [4] https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
# [5] https://www.iau.org/static/resolutions/IAU2012_English.pdf
# [6] de Pater, I., & Lissauer, J. J. (2015). Planetary Sciences (2nd ed.). Cambridge: Cambridge University Press.
# [7] https://ssd.jpl.nasa.gov/horizons/app.html#/

import numpy as np

from apygee.orbit import Orbit

G = 6.67428e-11  # [m^3 kg^-1 s^-2] [1]

AU = 149_597_870_700  # [m] [5]

M_SUN = 1_988_500e24  # [kg] [4]
R_SUN = 695_700e3  # [m] [4]
MU_SUN = G * M_SUN

M_MERCURY = 0.330103e24  # [kg] [2]
R_MERCURY = 2439.4e3  # [m] [2]
MU_MERCURY = G * M_MERCURY
A_MERCURY = 0.38709880 * AU  # [m] [6]
E_MERCURY = 0.20563175  # [6]
I_MERCURY = np.deg2rad(7.00499)  # [rad] [6]
OMEGA_MERCURY = np.deg2rad(48.3309)  # [rad] [6]
LOMEGA_MERCURY = np.deg2rad(77.4561)  # [rad] [6]
MERCURY = Orbit(
    [A_MERCURY, E_MERCURY, I_MERCURY, LOMEGA_MERCURY, OMEGA_MERCURY], mu=MU_SUN
)

M_VENUS = 4.86731e24  # [kg] [2]
R_VENUS = 6051.8e3  # [m] [2]
MU_VENUS = G * M_VENUS
A_VENUS = 0.72333201 * AU  # [m] [6]
E_VENUS = 0.00677177  # [6]
I_VENUS = np.deg2rad(3.39447)  # [6]
OMEGA_VENUS = np.deg2rad(76.6799)  # [6]
LOMEGA_VENUS = np.deg2rad(131.5637)  # [6]
VENUS = Orbit([A_VENUS, E_VENUS, I_VENUS, LOMEGA_VENUS, OMEGA_VENUS], mu=MU_SUN)

M_EARTH = 5.97217e24  # [kg] [2]
R_EARTH = 6371.0084e3  # [m] [2]
MU_EARTH = G * M_EARTH
A_EARTH = 1.00000083 * AU  # [m] [6]
E_EARTH = 0.016708617  # [6]
I_EARTH = np.deg2rad(0.0)  # [6]
OMEGA_EARTH = np.deg2rad(0.0)  # [6]
LOMEGA_EARTH = np.deg2rad(102.9374)  # [6]
EARTH = Orbit([A_EARTH, E_EARTH, I_EARTH, LOMEGA_EARTH, OMEGA_EARTH], mu=MU_SUN)

M_MOON = 0.07346e24  # [kg] [3]
R_MOON = 1738.1e3  # [m] [3]
MU_MOON = G * M_MOON
A_MOON = 384.40e6  # [m] [6]
E_MOON = 0.054900  # [6]
I_MOON = np.deg2rad(5.15)  # [6] (to ecliptic)
OMEGA_MOON = np.deg2rad(359.3)  # [7] (to ecliptic, mean on J2000)
LOMEGA_MOON = np.deg2rad(308.92)  # [7] (to ecliptic, mean on J2000)
MOON = Orbit([A_MOON, E_MOON, I_MOON, LOMEGA_MOON, OMEGA_MOON], mu=MU_EARTH)

M_MARS = 0.641691e24  # [kg] [2]
R_MARS = 3389.50e3  # [m] [2]
MU_MARS = G * M_MARS
A_MARS = 1.52368946 * AU  # [m] [6]
E_MARS = 0.09340062  # [6]
I_MARS = np.deg2rad(1.84973)  # [6]
OMEGA_MARS = np.deg2rad(49.5581)  # [6]
LOMEGA_MARS = np.deg2rad(336.6023)  # [6]
MARS = Orbit([A_MARS, E_MARS, I_MARS, LOMEGA_MARS, OMEGA_MARS], mu=MU_SUN)

M_JUPITER = 1898.125e24  # [kg] [2]
R_JUPITER = 69911e3  # [m] [2]
MU_JUPITER = G * M_JUPITER
A_JUPITER = 5.2027584 * AU  # [m] [6]
E_JUPITER = 0.048495  # [6]
I_JUPITER = np.deg2rad(1.3033)  # [6]
OMEGA_JUPITER = np.deg2rad(100.464)  # [6]
LOMEGA_JUPITER = np.deg2rad(14.331)  # [6]
JUPITER = Orbit(
    [A_JUPITER, E_JUPITER, I_JUPITER, LOMEGA_JUPITER, OMEGA_JUPITER], mu=MU_SUN
)

M_SATURN = 568.317e24  # [kg] [2]
R_SATURN = 58232e3  # [m] [2]
MU_SATURN = G * M_SATURN
A_SATURN = 9.5428244 * AU  # [m] [6]
E_SATURN = 0.055509  # [6]
I_SATURN = np.deg2rad(2.4889)  # [6]
OMEGA_SATURN = np.deg2rad(113.666)  # [6]
LOMEGA_SATURN = np.deg2rad(93.057)  # [6]
SATURN = Orbit([A_SATURN, E_SATURN, I_SATURN, LOMEGA_SATURN, OMEGA_SATURN], mu=MU_SUN)

M_URANUS = 86.8099e24  # [kg] [2]
R_URANUS = 25362e3  # [m] [2]
MU_URANUS = G * M_URANUS
A_URANUS = 19.19206 * AU  # [m] [6]
E_URANUS = 0.04630  # [6]
I_URANUS = np.deg2rad(0.773)  # [6]
OMEGA_URANUS = np.deg2rad(74.01)  # [6]
LOMEGA_URANUS = np.deg2rad(173.01)  # [6]
URANUS = Orbit([A_URANUS, E_URANUS, I_URANUS, LOMEGA_URANUS, OMEGA_URANUS], mu=MU_SUN)

M_NEPTUNE = 102.4092e24  # [kg] [2]
R_NEPTUNE = 24622e3  # [m] [2]
MU_NEPTUNE = G * M_NEPTUNE
A_NEPTUNE = 30.06893 * AU  # [m] [6]
E_NEPTUNE = 0.00899  # [6]
I_NEPTUNE = np.deg2rad(1.770)  # [6]
OMEGA_NEPTUNE = np.deg2rad(131.78)  # [6]
LOMEGA_NEPTUNE = np.deg2rad(48.12)  # [6]
NEPTUNE = Orbit(
    [A_NEPTUNE, E_NEPTUNE, I_NEPTUNE, LOMEGA_NEPTUNE, OMEGA_NEPTUNE], mu=MU_SUN
)
