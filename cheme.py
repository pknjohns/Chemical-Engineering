from __future__          import division      # Needed if running this with Python 2.7
import numpy             as np
import sympy             as sp
import matplotlib.pyplot as plt
import glob              as gb
import time
from scipy.optimize      import curve_fit
from scipy.optimize      import fsolve
from scipy.integrate     import odeint, quad
from scipy.interpolate   import interp1d
from scipy.misc          import derivative
from IPython.display     import Image
sp.init_printing()       # for pretty output for symbolic math
try:
    from IPython import get_ipython
    get_ipython().magic('matplotlib inline')
except (ImportError, NameError):
    pass  # If not in a Jupyter environment, ignore the magic command

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------

pi = np.pi
g = 9.8     # m/s^2, graviational accel constant
R = 8.314   # Gas constant, SI units

# -------------------------------------------------------------
# GENERAL USEFUL EQUATIONS
# -------------------------------------------------------------

def uc(str):
    """
    Function that provides factor to perform multiple unit
    conversions based on string input describing
    needed conversion
    """
    if (str=="m_to_ft"):
        return 3.2808           #ft
    if (str=="ft_to_m"):
        return 1/3.2808         #m
    if (str=="hr_to_s"):
        return 3600             #s
    if (str=="s_to_hr"):
        return 1/3600           #hr
    if (str=="kg_to_slug"):
        return 2.2046/32.174    #slug
    if (str=="slug_to_kg"):
        return 32.174/2.2046    #kg
    if (str=="K_to_degR"):
        return 1.8              #degR
    if (str=="degR_to_K"):
        return 1/1.8            #K
    
# ------------------------------------------------------------
# FLUIDS EQUATIONS
# ------------------------------------------------------------

def Re(rho,v,D,mu):
    """
    Function return Reynolds # based on density, velocity,
    pipe diameter, and viscosity
    """
    return rho*v*D/mu

def fhal(eps,D,Re):
    p = -1.8*np.log10((6.9/Re)+((eps/D)/3.7)**1.11)
    return 1/(p**2)

def Bern_t1(Vdot, P_t, D_guess, rho, mu, L, z):
    v2 = Vdot/(0.25*np.pi*D_guess**2)
    Re1 = rho*v2*D_guess/mu
    f_h1 = (-1.8*np.log10(6.9/Re1))**-2
    dPl = 0.5*rho*(L/D_guess)*f_h1*v2**2
    return P_t + rho*g*(z) - dPl

# -----------------------------------------------------------
# THERMO EQUATIONS OF STATE (EOS)
# -----------------------------------------------------------

def acentric(Pr):
    """
    Function to calculate acentric factor for a material
    at Tr = 0.7
    """
    return -1.0 - np.log10(Pr)

def pitzer(Pr, Tr, omega):
    """
    Pitzer Correlation, truncated, virial EOS
    Function used to find compressibility factor Z
    """
    b0 = 0.083 - 0.422/(Tr**1.6)
    b1 = 0.139 - 0.172/(Tr**4.2)
    return 1 + b0*Pr/Tr + omega*b1*Pr/Tr

