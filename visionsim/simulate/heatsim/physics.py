"""Physical constants in SI and solver millimetre units."""

import math

STEFAN_BOLTZMANN_SI = 5.670374419e-8  # W/(m² K⁴)
STEFAN_BOLTZMANN_MM = STEFAN_BOLTZMANN_SI / 1e6
CYCLES_LOUT_TO_IRRADIANCE = math.pi
