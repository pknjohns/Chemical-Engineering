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

def T_red(T, Tc):
    """
    Function to calculate the reduced temp of a mtrl
    T = temp of mtrl
    Tc = critical temp of mtrl
    """
    return T/Tc

def P_red(P, Pc):
    """
    Function to calculate the reduced pressure of a mtrl
    P = pressure of mtrl
    Pc = critical pressure of mtrl
    """
    return P/Pc

def compress_factor(P, V, T):
    """
    Function to calculate compressibility factor for non-ideal gases
    V is molar volume of the gas
    P is gas pressure
    T is gas temperature
    """
    return P*V/R/T

def acentric(Pr):
    """
    Function to calculate acentric factor (omega) for a material
    at Tr = 0.7
    """
    return -1.0 - np.log10(Pr)

def pitzer(Pr, Tr, omega):
    """
    Pitzer Correlation, truncated, virial EOS
    Function used to find compressibility factor Z

    Pr is the reduced pressure
    Tr is the reduced temperature
    omega is the acentric factor
    b0 and b1 are the virial coefficients, which are both
    functions of temp
    simple to solve & reasonably accurate
    """
    b0 = 0.083 - 0.422/(Tr**1.6)
    b1 = 0.139 - 0.172/(Tr**4.2)
    return 1 + b0*Pr/Tr + omega*b1*Pr/Tr

"""
General Cubic EOS form:

Z = 1 + Beta - q*Beta*((Z-Beta)/((Z + eps*Beta)*(Z + sigma*Beta)))

Beta = Omega*Pr/Tr

q = (Phi*alpha(Tr))/(Omega*Tr)

Omega, Phi, eps, sigma are constants defined depending on EOS used
alpha(Tr) means alpha is a function of Tr and the acentric factor (omega)

Use to solve for molar volume (Vm)
    largest Vm = gas
    smallest Vm = liquid
    middle Vm = no pertinent meaning
"""

def cubEOS(Z, alpha, sig, eps, Ome, Phi, Tr, Pr):
    Beta = Ome*Pr/Tr
    q = Phi*alpha/Ome/Tr
    return 1 + Beta - q*Beta*((Z-Beta)/((Z + eps*Beta)*(Z + sig*Beta))) - Z

def vdw(V, Tc, Pc, T, P):
    """
    Van der Waals EOS
    Used to describe non-ideal behavior of gasses
    Can use to find molar volume with fsolve
    Tc = critical temp of mtrl
    Pc = critical pressure of mtrl
    V = guess for molar volume
    T = temp of mtrl
    P = pressure of mtrl
    """
    a = (27/64)*(R**2)*(Tc**2)/Pc
    b = (1/8)*(R*Tc/Pc)

    return (R*T)/(V-b) - a/(V**2) - P

def rk(Z, Tr, Pr):
    """
    Redlich Kwong cubic EOS
    Used to describe non-ideal behavior of gasses
    Can use to find molar volume with fsolve
    """
    alpha = Tr**-0.5
    sig = 1
    eps = 0
    Ome = 0.08664
    Phi = 0.42748
    return cubEOS(Z, alpha, sig, eps, Ome, Phi, Tr, Pr)

def srk(Z, Tr, Pr, ome):
    """
    Soave Redlich Kwong cubic EOS
    Used to describe non-ideal behavior of gasses
    Can use to find molar volume with fsolve
    Very accurate for broad range of conditions
    """
    alpha = (1 + (0.480 + 1.574*ome - 0.176*ome**2)*(1-Tr**0.5))**2
    sig = 1
    eps = 0
    Ome = 0.08664
    Phi = 0.42748
    return cubEOS(Z, alpha, sig, eps, Ome, Phi, Tr, Pr)

def pr(Z, Tr, Pr, ome):
    """
    Perg Robinson cubic EOS
    Used to describe non-ideal behavior of gasses
    Can use to find molar volume with fsolve
    Very accurate for broad range of conditions (favored among EOS)
    """
    alpha = (1 + (0.37464 + 1.54226*ome - 0.26992*ome**2)*(1 - Tr**0.5))**2
    sig = 1 + np.sqrt(2)
    eps = 1 - np.sqrt(2)
    Ome = 0.07780
    Phi = 0.45724
    return cubEOS(Z, alpha, sig, eps, Ome, Phi, Tr, Pr)