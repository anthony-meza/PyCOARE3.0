import numpy as np

def psit_26(zet):
    # Computes temperature structure function
    dzet = np.minimum(50, 0.35 * zet)  # Stable
    psi = -((1 + 0.6667 * zet) ** 1.5 + 0.6667 * (zet - 14.28) * np.exp(-dzet) + 8.525)
    k = np.where(zet < 0)  # Unstable
    x = (1 - 15 * zet[k]) ** 0.5
    psik = 2 * np.log((1 + x) / 2)
    x = (1 - 34.15 * zet[k]) ** 0.3333
    psic = 1.5 * np.log((1 + x + x ** 2) / 3) - np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3)) + 4 * np.arctan(1) / np.sqrt(3)
    f = zet[k] ** 2 / (1 + zet[k] ** 2)
    psi[k] = (1 - f) * psik + f * psic
    return psi

def psiu_26(zet):
    # Computes velocity structure function
    dzet = np.minimum(50, 0.35 * zet)  # Stable
    psi = -((1 + zet) + 0.6667 * (zet - 14.28) * np.exp(-dzet) + 8.525)
    k = np.where(zet < 0)  # Unstable
    x = (1 - 15 * zet[k]) ** 0.25
    psik = 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + 2 * np.arctan(1)
    x = (1 - 10.15 * zet[k]) ** 0.3333
    psic = 1.5 * np.log((1 + x + x ** 2) / 3) - np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3)) + 4 * np.arctan(1) / np.sqrt(3)
    f = zet[k] ** 2 / (1 + zet[k] ** 2)
    psi[k] = (1 - f) * psik + f * psic
    return psi

def bucksat(T, P):
    # Computes saturation vapor pressure [mb] given T [degC] and P [mb]
    exx = 6.1121 * np.exp(17.502 * T / (T + 240.97)) * (1.0007 + 3.46e-6 * P)
    return exx

def qsat26sea(T, P):
    # Computes surface saturation specific humidity [g/kg] given T [degC] and P [mb]
    ex = bucksat(T, P)
    es = 0.98 * ex  # Reduction at sea surface
    qs = 622 * es / (P - 0.378 * es)
    return qs

def qsat26air(T, P, rh):
    # Computes specific humidity [g/kg] given T [degC], RH [%], and P [mb]
    es = bucksat(T, P)
    em = 0.01 * rh * es
    qs = 622 * em / (P - 0.378 * em)
    return qs

def grv(lat):
    # Computes g [m/s^2] given lat in degrees
    gamma = 9.7803267715
    c1 = 0.0052790414
    c2 = 0.0000232718
    c3 = 0.0000001262
    c4 = 0.0000000007
    phi = lat * np.pi / 180
    x = np.sin(phi)
    g = gamma * (1 + c1 * x**2 + c2 * x**4 + c3 * x**6 + c4 * x**8)
    return g

def RHcalc(T, P, Q):
    # Computes relative humidity given T, P, and Q
    es = 6.1121 * np.exp(17.502 * T / (T + 240.97)) * (1.0007 + 3.46e-6 * P)
    em = Q * P / (0.378 * Q + 0.622)
    RHrf = 100 * em / es
    return RHrf
def is_float(variable):
    return isinstance(variable, float)
def coare30vncoare30vn(u, zu, t, zt, rh, zq, P=1015., ts=25., Rs=150., Rl=370., lats=45., zi=600., zref_u=3., zref_t=3., zref_q=3., nits = 6, jcool = 0):
    """
    WHOI Vectorized COARE 3.0a

    Vectorized version of COARE3 code (Fairall et al, 2003) with cool skin option retained but warm layer and surface wave
    options removed. Assumes u, t, rh, ts are vectors; sensor heights zu, zt, zl, latitude lat, and PBL height zi are constants;
    air pressure P and radiation Rs, Rl may be vectors or constants. Default values are assigned for P, Rs, Rl, lat, and zi if these
    data are not available. Defaults should be set to representative regional values if possible.

    Args:
        u (array-like): Relative wind speed (m/s) at height zu (m).
        zu (float): Sensor height for wind (m).
        t (array-like): Bulk air temperature (degC) at height zt (m).
        zt (float): Sensor height for air temperature (m).
        rh (array-like): Relative humidity (%) at height zq (m).
        zq (float): Sensor height for relative humidity (m).
        P (float, optional): Surface air pressure (mb) (default = 1015).
        ts (float, optional): Water temperature (degC) (default = 25).
        Rs (float or array-like, optional): Downward shortwave radiation (W/m^2) (default = 150).
        Rl (float or array-like, optional): Downward longwave radiation (W/m^2) (default = 370).
        lat (float, optional): Latitude (default = +45 N).
        zi (float, optional): Planetary Boundary Layer (PBL) height (m) (default = 600m).
        zref_u (float, optional): Reference height for wind (m) (default = 3).
        zref_t (float, optional): Reference height for air temperature (m) (default = 3).
        zref_q (float, optional): Reference height for specific humidity (m) (default = 3).
        U10n (array-like, optional): Wind speed at reference height, zref_u (m/s).

    # Set jcool=1 if Ts is bulk ocean temperature (default),
    # jcool=0 if Ts is true ocean skin temperature.
    Returns:
        dict: A dictionary containing the following output parameters:
            - 'usr' (float): Friction velocity (m/s).
            - 'tau' (float): Wind stress (N/m^2).
            - 'hsb' (float): Sensible heat flux into the ocean (W/m^2).
            - 'hlb' (float): Latent heat flux into the ocean (W/m^2).
            - 'hbb' (float): Buoyancy flux into the ocean (W/m^2).
            - 'hsbb' (float): Sonic buoyancy flux into the ocean (W/m^2).
            - 'tsr' (float): t* (temperature scale parameter).
            - 'qsr' (float): q* (humidity scale parameter).
            - 'zot' (float): z_o for temperature (m).
            - 'zoq' (float): z_o for humidity (m).
            - 'Cd' (float): Wind stress transfer coefficient at height zu.
            - 'Ch' (float): Sensible heat transfer coefficient at height zt.
            - 'Ce' (float): Latent heat transfer coefficient at height zq.
            - 'L' (float): Obukhov length scale (m).
            - 'zet' (float): Monin-Obukhov stability parameter zu/L.
            - 'dter' (float): Cool-skin temperature depression (degC).
            - 'tkt' (float): Cool-skin thickness (m).
            - 'Urf' (array-like): Wind speed at reference height, zref_u (m/s).
            - 'Trf' (array-like): Temperature at reference height, zref_t (C).
            - 'Qrf' (array-like): Specific humidity at reference height, zref_q (g/kg).
            - 'RHrf' (array-like): Relative humidity at reference height (%).
    """

    zu = np.ones_like(u) * zu
    t = np.ones_like(u) * t
    zt = np.ones_like(u) * zt
    rh = np.ones_like(u) * rh
    zq = np.ones_like(u) * zq
    P = np.ones_like(u) * P
    ts = np.ones_like(u) * ts
    Rs = np.ones_like(u) * Rs
    Rl = np.ones_like(u) * Rl
    lats = np.ones_like(u) * lats
    zi = np.ones_like(u) * zi
    zref_u = np.ones_like(u) * zref_u
    zref_t = np.ones_like(u) * zref_t
    zref_q = np.ones_like(u) * zref_q


    # Input variable u is assumed relative wind speed
    us = np.zeros_like(u)
    # Convert rh to specific humidity
    Qs = qsat26sea(ts, P) / 1000  # surface water specific humidity (g/kg)
    Q = qsat26air(t, P, rh) / 1000  # specific humidity of air (g/kg)

    # Set rain to zero
    rain = np.zeros_like(u)

    # Set constants
    Beta = 1.2
    von = 0.4
    fdg = 1.00
    tdk = 273.16
    lat = np.deg2rad(lats)
    grav = 9.81

    # Air constants
    Rgas = 287.1
    Le = (2.501 - 0.00237 * ts) * 1e6
    cpa = 1004.67
    cpv = cpa * (1 + 0.84 * Q)
    rhoa = P * 100 / (Rgas * (t + tdk) * (1 + 0.61 * Q))
    visa = 1.326e-5 * (1 + 6.542e-3 * t + 8.301e-6 * t**2 - 4.84e-9 * t**3)

    # Cool skin constants
    Al = 2.1e-5 * (ts + 3.2)**0.79
    be = 0.026
    cpw = 4000
    rhow = 1022
    visw = 1e-6
    tcw = 0.6
    bigc = 16 * grav * cpw * (rhow * visw)**3 / (tcw**2 * rhoa**2)
    wetc = 0.622 * Le * Qs / (Rgas * (ts + tdk)**2)

    # Net radiation fluxes
    Rns = 0.945 * Rs  # albedo correction
    Rnl = 0.97 * (5.67e-8 * (ts - 0.3 * jcool + tdk)**4 - Rl)  # initial value

    # First guess
    du = u - us
    dt = ts - t - 0.0098 * zt
    dq = Qs - Q
    ta = t + tdk
    ug = 0.5
    dter = 0.3
    ut = np.sqrt(du**2 + ug**2)
    u10 = ut * np.log(10 / 1e-4) / np.log(zu / 1e-4)
    usr = 0.035 * u10
    zo10 = 0.011 * usr**2 / grav + 0.11 * visa / usr
    Cd10 = (von / np.log(10 / zo10))**2
    Ch10 = 0.00115
    Ct10 = Ch10 / np.sqrt(Cd10)
    zot10 = 10 / np.exp(von / Ct10)
    Cd = (von / np.log(zu / zo10))**2 
    Ct = von / np.log(zt / zot10)
    CC = von * Ct / Cd
    Ribcu = -zu / zi / 0.004 / Beta**3
    Ribu = -grav * zu / ta * ((dt - dter * jcool) + 0.61 * ta * dq) / ut**2
    zetu = CC * Ribu * (1 + 27 / 9 * Ribu / CC) 
    k50 = np.where(zetu > 50)  # stable with very thin M-O length relative to zu
    k = np.where(Ribu < 0)
    zetu[k] = CC[k] * Ribu[k] / (1 + Ribu[k] / Ribcu[k])

    L10 = zu / zetu
    usr = ut * von / (np.log(zu / zo10) - psiu_26(zu / L10))
    tsr = -(dt - dter * jcool) * von * fdg / (np.log(zt / zot10) - psit_26(zt / L10))
    qsr = -(dq - wetc * dter * jcool) * von * fdg / (np.log(zq / zot10) - psit_26(zq / L10))
    tkt = 0.001 * np.ones_like(u)
    charn = 0.011 * np.ones_like(u)
    k = np.where(ut > 10)
    charn[k] = 0.011 + (ut[k] - 10) / (18 - 10) * (0.018 - 0.011)
    k = np.where(ut > 18)
    charn[k] = 0.018
    # number of iterations

    # Bulk loop
    for i in range(nits):
        zet = von * grav * zu / ta * (tsr + 0.61 * ta * qsr) / (usr**2)
        zo = charn * usr**2 / grav + 0.11 * visa / usr  # surface roughness
        rr = zo * usr / visa
        L = zu / zet
        zoq = np.minimum(1.15e-4, 5.5e-5 / rr**0.6)  # moisture roughness
        zot = zoq  # temperature roughness
        usr = ut * von / (np.log(zu / zo) - psiu_26(zu / L))
        tsr = -(dt - dter * jcool) * von * fdg / (np.log(zt / zot) - psit_26(zt / L))
        qsr = -(dq - wetc * dter * jcool) * von * fdg / (np.log(zq / zoq) - psit_26(zq / L))
        tvsr = tsr + 0.61 * ta * qsr
        tssr = tsr + 0.51 * ta * qsr
        Bf = -grav / ta * usr * tvsr
        ug = 0.2 * np.ones_like(u)
        k = np.where(Bf > 0)
        ug[k] = Beta * (Bf[k] * zi[k])**0.333
        ut = np.sqrt(du**2 + ug**2)
        hsb = -rhoa * cpa * usr * tsr
        hlb = -rhoa * Le * usr * qsr
        qout = Rnl + hsb + hlb
        dels = Rns * (0.065 + 11 * tkt - 6.6e-5 / tkt * (1 - np.exp(-tkt / 8.0e-4)))
        qcol = qout - dels
        alq = Al * qcol + be * hlb * cpw / Le
        xlamx = 6.0 * np.ones_like(u)
        tkt = np.minimum(0.01, xlamx * visw / (np.sqrt(rhoa / rhow) * usr))
        k = np.where(alq > 0)
        xlamx[k] = 6 / (1 + (bigc[k] * alq[k] / usr[k]**4)**0.75)**0.333
        tkt[k] = xlamx[k] * visw / (np.sqrt(rhoa[k] / rhow) * usr[k])
        dter = qcol * tkt / tcw
        dqer = wetc * dter
        Rnl = 0.97 * (5.67e-8 * (ts - dter * jcool + tdk)**4 - Rl)  # update dter
        if i == 0:  # save first iteration solution for case of zetu > 50
            usr50 = usr[k50]
            tsr50 = tsr[k50]
            qsr50 = qsr[k50]
            L50 = L[k50]
            zet50 = zet[k50]
            dter50 = dter[k50]
            dqer50 = dqer[k50]
            tkt50 = tkt[k50]

    # Insert first iteration solution for case with zetu > 50
    usr[k50] = usr50
    tsr[k50] = tsr50
    qsr[k50] = qsr50
    L[k50] = L50
    zet[k50] = zet50
    dter[k50] = dter50
    dqer[k50] = dqer50
    tkt[k50] = tkt50

    # Compute fluxes
    tau = rhoa * usr**2 * du / ut  # wind stress
    hsb = rhoa * cpa * usr * tsr  # sensible heat flux
    hlb = rhoa * Le * usr * qsr  # latent heat flux
    hbb = rhoa * cpa * usr * tvsr  # buoyancy flux
    hsbb = rhoa * cpa * usr * tssr  # sonic heat flux

    # Compute transfer coefficients relative to ut @ meas. height
    Cd = tau / rhoa / ut / np.maximum(0.1, du)
    Ch = -usr * tsr / ut / (dt - dter * jcool)
    Ce = -usr * qsr / (dq - dqer * jcool) / ut

    # Compute 10-m neutral coefficients relative to ut
    Urf = u + usr / von * (np.log(zref_u / zu) - psiu_26(zref_u / L) + psiu_26(zu / L))
    Trf = t + tsr / von / fdg * (np.log(zref_t / zt) - psit_26(zref_t / L) + psit_26(zt / L)) + 0.0098 * (zt - zref_t)
    Qrf = Q + qsr / von / fdg * (np.log(zref_q / zq) - psit_26(zref_q / L) + psit_26(zq / L))
    RHrf = RHcalc(Trf, P, Qrf)
    Qrf = Qrf * 1000
    # U10n = u + usr / von * (np.log(10 / zu) + psiu_26(zu / L))

    # Output
    A = {
        "usr": usr,
        "tau": tau,
        "hsb": hsb,
        "hlb": hlb,
        "hbb": hbb,
        "hsbb": hsbb,
        "tsr": tsr,
        "qsr": qsr,
        "zo": zo,
        "zot": zot,
        "zoq": zoq,
        "Cd": Cd,
        "Ch": Ch,
        "Ce": Ce,
        "L": L,
        "zet": zet,
        "dter": dter,
        "tkt": tkt,
        "Urf": Urf,
        "Trf": Trf,
        "Qrf": Qrf,
        "RHrf": RHrf
    }
    return A
