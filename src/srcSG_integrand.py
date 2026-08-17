import torch
import math
import numpy as np
from srcSG_utilities import  _to_tensor, to_tensor

def fden_3D_exact(x: torch.Tensor, param: dict):
    device, dtype = x.device, x.dtype
    Nd = param['nd']

    # ---- domain transform ----
    a1 = torch.tensor(param['xmin'], dtype=dtype, device=device)[:Nd]
    b1 = torch.tensor(param['xmax'], dtype=dtype, device=device)[:Nd]
    x_phys = (b1 - a1) * x + a1
    vol = torch.prod(b1 - a1)

    # ---- parameters ----
    b  = torch.tensor(param['b'], dtype=dtype, device=device)
    R  = torch.tensor(param['R'], dtype=dtype, device=device)
    Rm = torch.tensor(param['Rm'], dtype=dtype, device=device)

    x, y, z = x_phys[..., 0], x_phys[..., 1], x_phys[..., 2]    

    # Common constants
    sqrt2 = math.sqrt(2.0)
    sqrt6 = math.sqrt(6.0)
    pi = math.pi
    sqrt5_19 = math.sqrt(5.0 / 19.0)
    #const1 = 40.0 * sqrt5_19 / (19.0 * (pi ** 1.5))
    
    log_const1 = math.log(40.0 * sqrt5_19 / (19.0 * (pi ** 1.5)))
    log_const1 = torch.tensor(log_const1, dtype=dtype, device=device)

    b2 = b * b
    inv_b2 = 1.0 / b2

    # ---------- numerator factor (common exponential) ----------
    num_factor_exp = - (1420.0 * R**2 - 6840.0 * Rm * R + 297.0 * Rm**2 +
                        1800.0 * (x**2 + y**2 + z**2)) / (1710.0 * b2)
    num_factor = torch.exp(num_factor_exp)

    # ---------- bracket sum (huge sum of exponentials) ----------
    # We'll accumulate the sum using a list of (coeff, denominator D, polynomial)
    # where the exponential is exp(poly / (D * b2))

    # Helper to compute polynomial for a given term; many terms share structure.
    # We'll directly compute each term using the explicit formulas from the LaTeX.

    #bracket_sum = torch.zeros_like(R)
    bracket_sum = torch.zeros_like(x)

    # ---- first group (denominator 342) ----
    # term1: -6 * exp( (194 R^2 - 24(43 Rm -15z)R + 27 Rm(Rm-8z)) / (342 b2) )
    poly1 = 194.0 * R**2 - 24.0 * (43.0 * Rm - 15.0 * z) * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -6.0 * torch.exp(poly1 / (342.0 * b2))

    # term2: +6 * exp( (422 R^2 - 24(43 Rm -15z)R + 27 Rm(Rm-8z)) / (342 b2) )
    poly2 = 422.0 * R**2 - 24.0 * (43.0 * Rm - 15.0 * z) * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += 6.0 * torch.exp(poly2 / (342.0 * b2))

    # term3: +4 * exp( (346 R^2 - 180(7 Rm -2z)R + 27 Rm(Rm-8z)) / (342 b2) )
    poly3 = 346.0 * R**2 - 180.0 * (7.0 * Rm - 2.0 * z) * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += 4.0 * torch.exp(poly3 / (342.0 * b2))

    # term4: -2 * exp( (574 R^2 - 180(7 Rm -2z)R + 27 Rm(Rm-8z)) / (342 b2) )
    poly4 = 574.0 * R**2 - 180.0 * (7.0 * Rm - 2.0 * z) * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -2.0 * torch.exp(poly4 / (342.0 * b2))

    # term5: -2 * exp( (802 R^2 - 180(7 Rm -2z)R + 27 Rm(Rm-8z)) / (342 b2) )
    poly5 = 802.0 * R**2 - 180.0 * (7.0 * Rm - 2.0 * z) * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -2.0 * torch.exp(poly5 / (342.0 * b2))

    # term6: +3 * exp( (379 R^2 - 1026 Rm R - 576 Rm z) / (342 b2) )
    poly6 = 379.0 * R**2 - 1026.0 * Rm * R - 576.0 * Rm * z
    bracket_sum += 3.0 * torch.exp(poly6 / (342.0 * b2))

    # term7: -5 * exp( (607 R^2 - 1026 Rm R - 576 Rm z) / (342 b2) )
    poly7 = 607.0 * R**2 - 1026.0 * Rm * R - 576.0 * Rm * z
    bracket_sum += -5.0 * torch.exp(poly7 / (342.0 * b2))

    # term8: +1 * exp( (835 R^2 - 1026 Rm R - 576 Rm z) / (342 b2) )
    poly8 = 835.0 * R**2 - 1026.0 * Rm * R - 576.0 * Rm * z
    bracket_sum += torch.exp(poly8 / (342.0 * b2))

    # term9: +1 * exp( (1063 R^2 - 1026 Rm R - 576 Rm z) / (342 b2) )
    poly9 = 1063.0 * R**2 - 1026.0 * Rm * R - 576.0 * Rm * z
    bracket_sum += torch.exp(poly9 / (342.0 * b2))

    # ---- group with denominator 114 ----
    # term10: +3 * exp( (15 Rm^2 - 352 R Rm + 48 z Rm + 240 R z) / (114 b2) )
    poly10 = 15.0 * Rm**2 - 352.0 * R * Rm + 48.0 * z * Rm + 240.0 * R * z
    bracket_sum += 3.0 * torch.exp(poly10 / (114.0 * b2))

    # term11: -3 * exp( (76 R^2 - 352 Rm R + 240 z R + 15 Rm^2 + 48 Rm z) / (114 b2) )
    poly11 = 76.0 * R**2 - 352.0 * Rm * R + 240.0 * z * R + 15.0 * Rm**2 + 48.0 * Rm * z
    bracket_sum += -3.0 * torch.exp(poly11 / (114.0 * b2))

    # ---- terms with sqrt2 and sqrt6 (denominator 342) ----
    # term12: +2 * exp( (259 R^2 - 6(179 Rm + 20(-2z + y√2 + x√6))R + 72 Rm(3Rm+2z)) / (342 b2) )
    inner12 = 179.0 * Rm + 20.0 * (-2.0 * z + y * sqrt2 + x * sqrt6)
    poly12 = 259.0 * R**2 - 6.0 * inner12 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += 2.0 * torch.exp(poly12 / (342.0 * b2))

    # term13: -2 * exp( (487 R^2 - 6(179 Rm + 20(-2z + y√2 + x√6))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly13 = 487.0 * R**2 - 6.0 * inner12 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly13 / (342.0 * b2))

    # term14: -2 * exp( (247 R^2 - 90(13 Rm - 8z)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly14 = 247.0 * R**2 - 90.0 * (13.0 * Rm - 8.0 * z) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly14 / (342.0 * b2))

    # term15: +1 * exp( (475 R^2 - 90(13 Rm - 8z)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly15 = 475.0 * R**2 - 90.0 * (13.0 * Rm - 8.0 * z) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += torch.exp(poly15 / (342.0 * b2))

    # term16: +1 * exp( (703 R^2 - 90(13 Rm - 8z)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly16 = 703.0 * R**2 - 90.0 * (13.0 * Rm - 8.0 * z) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += torch.exp(poly16 / (342.0 * b2))

    # term17: +2 * exp( (259 R^2 - 978 Rm R + 120(-2z + y√2 + x√6)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly17 = 259.0 * R**2 - 978.0 * Rm * R + 120.0 * (-2.0 * z + y * sqrt2 + x * sqrt6) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += 2.0 * torch.exp(poly17 / (342.0 * b2))

    # term18: -2 * exp( (487 R^2 - 978 Rm R + 120(-2z + y√2 + x√6)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly18 = 487.0 * R**2 - 978.0 * Rm * R + 120.0 * (-2.0 * z + y * sqrt2 + x * sqrt6) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly18 / (342.0 * b2))

    # term19: +2 * exp( (259 R^2 - 1074 Rm R + 240(z + y√2)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly19 = 259.0 * R**2 - 1074.0 * Rm * R + 240.0 * (z + y * sqrt2) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += 2.0 * torch.exp(poly19 / (342.0 * b2))

    # term20: -2 * exp( (487 R^2 - 1074 Rm R + 240(z + y√2)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly20 = 487.0 * R**2 - 1074.0 * Rm * R + 240.0 * (z + y * sqrt2) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly20 / (342.0 * b2))

    # term21: +2 * exp( (259 R^2 - 1074 Rm R + 120(2z - √2 y + x√6)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly21 = 259.0 * R**2 - 1074.0 * Rm * R + 120.0 * (2.0 * z - sqrt2 * y + x * sqrt6) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += 2.0 * torch.exp(poly21 / (342.0 * b2))

    # term22: -2 * exp( (487 R^2 - 1074 Rm R + 120(2z - √2 y + x√6)R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly22 = 487.0 * R**2 - 1074.0 * Rm * R + 120.0 * (2.0 * z - sqrt2 * y + x * sqrt6) * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly22 / (342.0 * b2))

    # ---- denominator 114 terms ----
    # term23: +2 * exp( (80 R^2 - 288 Rm R + 40(-2z + y√2 + x√6)R + 3 Rm(5Rm+16z)) / (114 b2) )
    poly23 = 80.0 * R**2 - 288.0 * Rm * R + 40.0 * (-2.0 * z + y * sqrt2 + x * sqrt6) * R + 3.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly23 / (114.0 * b2))

    # term24: +2 * exp( (80 R^2 - 8(36 Rm -5√2 y +10z +5x√6)R + 3 Rm(5Rm+16z)) / (114 b2) )
    inner24 = 36.0 * Rm - 5.0 * sqrt2 * y + 10.0 * z + 5.0 * x * sqrt6
    poly24 = 80.0 * R**2 - 8.0 * inner24 * R + 3.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly24 / (114.0 * b2))

    # ---- more denominator 342 terms ----
    # term25: -2 * exp( (164 R^2 - 12(99 Rm + 10(-2z + y√2 + x√6))R + 9 Rm(5Rm+16z)) / (342 b2) )
    inner25 = 99.0 * Rm + 10.0 * (-2.0 * z + y * sqrt2 + x * sqrt6)
    poly25 = 164.0 * R**2 - 12.0 * inner25 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -2.0 * torch.exp(poly25 / (342.0 * b2))

    # term26: +2 * exp( (392 R^2 - 12(99 Rm + 10(-2z + y√2 + x√6))R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly26 = 392.0 * R**2 - 12.0 * inner25 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly26 / (342.0 * b2))

    # term27: -4 * exp( (164 R^2 - 1092 Rm R + 120(-2z + y√2 + x√6)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly27 = 164.0 * R**2 - 1092.0 * Rm * R + 120.0 * (-2.0 * z + y * sqrt2 + x * sqrt6) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -4.0 * torch.exp(poly27 / (342.0 * b2))

    # term28: -2 * exp( (164 R^2 - 1188 Rm R + 240(z + y√2)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly28 = 164.0 * R**2 - 1188.0 * Rm * R + 240.0 * (z + y * sqrt2) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -2.0 * torch.exp(poly28 / (342.0 * b2))

    # term29: +2 * exp( (392 R^2 - 1188 Rm R + 240(z + y√2)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly29 = 392.0 * R**2 - 1188.0 * Rm * R + 240.0 * (z + y * sqrt2) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly29 / (342.0 * b2))

    # term30: -1 * exp( (304 R^2 - 120(11 Rm -4√2 y + 2z)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly30 = 304.0 * R**2 - 120.0 * (11.0 * Rm - 4.0 * sqrt2 * y + 2.0 * z) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -torch.exp(poly30 / (342.0 * b2))

    # term31: -1 * exp( (532 R^2 - 120(11 Rm -4√2 y + 2z)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly31 = 532.0 * R**2 - 120.0 * (11.0 * Rm - 4.0 * sqrt2 * y + 2.0 * z) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -torch.exp(poly31 / (342.0 * b2))

    # term32: +2 * exp( (316 R^2 - 120(11 Rm - √6 x - √2 y + 2z)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly32 = 316.0 * R**2 - 120.0 * (11.0 * Rm - sqrt6 * x - sqrt2 * y + 2.0 * z) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly32 / (342.0 * b2))

    # term33: -2 * exp( (164 R^2 - 1188 Rm R + 120(2z - √2 y + x√6)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly33 = 164.0 * R**2 - 1188.0 * Rm * R + 120.0 * (2.0 * z - sqrt2 * y + x * sqrt6) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -2.0 * torch.exp(poly33 / (342.0 * b2))

    # term34: +2 * exp( (392 R^2 - 1188 Rm R + 120(2z - √2 y + x√6)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly34 = 392.0 * R**2 - 1188.0 * Rm * R + 120.0 * (2.0 * z - sqrt2 * y + x * sqrt6) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly34 / (342.0 * b2))

    # term35: +2 * exp( (316 R^2 - 120(11 Rm - √2 y + 2z + x√6)R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly35 = 316.0 * R**2 - 120.0 * (11.0 * Rm - sqrt2 * y + 2.0 * z + x * sqrt6) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly35 / (342.0 * b2))

    # term36: -2 * exp( (76 R^2 - 16(18 Rm + 5(z - 2√2 y))R + 3 Rm(5Rm+16z)) / (114 b2) )
    inner36 = 18.0 * Rm + 5.0 * (z - 2.0 * sqrt2 * y)
    poly36 = 76.0 * R**2 - 16.0 * inner36 * R + 3.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -2.0 * torch.exp(poly36 / (114.0 * b2))

    # term37: -2 * exp( (346 R^2 - 24(49 Rm + 5(z - 2√2 y))R + 27 Rm(Rm-8z)) / (342 b2) )
    inner37 = 49.0 * Rm + 5.0 * (z - 2.0 * sqrt2 * y)
    poly37 = 346.0 * R**2 - 24.0 * inner37 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -2.0 * torch.exp(poly37 / (342.0 * b2))

    # term38: +2 * exp( (574 R^2 - 24(49 Rm + 5(z - 2√2 y))R + 27 Rm(Rm-8z)) / (342 b2) )
    poly38 = 574.0 * R**2 - 24.0 * inner37 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += 2.0 * torch.exp(poly38 / (342.0 * b2))

    # term39: +2 * exp( (422 R^2 - 12(79 Rm + 10(z - 2√2 y))R + 27 Rm(Rm-8z)) / (342 b2) )
    inner39 = 79.0 * Rm + 10.0 * (z - 2.0 * sqrt2 * y)
    poly39 = 422.0 * R**2 - 12.0 * inner39 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += 2.0 * torch.exp(poly39 / (342.0 * b2))

    # term40: -2 * exp( (650 R^2 - 12(79 Rm + 10(z - 2√2 y))R + 27 Rm(Rm-8z)) / (342 b2) )
    poly40 = 650.0 * R**2 - 12.0 * inner39 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -2.0 * torch.exp(poly40 / (342.0 * b2))

    # term41: +4 * exp( (152 R^2 - 12(91 Rm + 20(z - 2√2 y))R + 9 Rm(5Rm+16z)) / (342 b2) )
    inner41 = 91.0 * Rm + 20.0 * (z - 2.0 * sqrt2 * y)
    poly41 = 152.0 * R**2 - 12.0 * inner41 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 4.0 * torch.exp(poly41 / (342.0 * b2))

    # term42: -2 * exp( (247 R^2 - 6(163 Rm + 40(z - 2√2 y))R + 72 Rm(3Rm+2z)) / (342 b2) )
    inner42 = 163.0 * Rm + 40.0 * (z - 2.0 * sqrt2 * y)
    poly42 = 247.0 * R**2 - 6.0 * inner42 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly42 / (342.0 * b2))

    # term43: +1 * exp( (475 R^2 - 6(163 Rm + 40(z - 2√2 y))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly43 = 475.0 * R**2 - 6.0 * inner42 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += torch.exp(poly43 / (342.0 * b2))

    # term44: +1 * exp( (703 R^2 - 6(163 Rm + 40(z - 2√2 y))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly44 = 703.0 * R**2 - 6.0 * inner42 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += torch.exp(poly44 / (342.0 * b2))

    # term45: +2 * exp( (316 R^2 - 120(11 Rm + 2(z + y√2))R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly45 = 316.0 * R**2 - 120.0 * (11.0 * Rm + 2.0 * (z + y * sqrt2)) * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly45 / (342.0 * b2))

    # term46: +2 * exp( (80 R^2 - 16(18 Rm + 5(z + y√2))R + 3 Rm(5Rm+16z)) / (114 b2) )
    inner46 = 18.0 * Rm + 5.0 * (z + y * sqrt2)
    poly46 = 80.0 * R**2 - 16.0 * inner46 * R + 3.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 2.0 * torch.exp(poly46 / (114.0 * b2))

    # term47: -4 * exp( (164 R^2 - 12(91 Rm + 20(z + y√2))R + 9 Rm(5Rm+16z)) / (342 b2) )
    inner47 = 91.0 * Rm + 20.0 * (z + y * sqrt2)
    poly47 = 164.0 * R**2 - 12.0 * inner47 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -4.0 * torch.exp(poly47 / (342.0 * b2))

    # term48: +2 * exp( (259 R^2 - 6(163 Rm + 40(z + y√2))R + 72 Rm(3Rm+2z)) / (342 b2) )
    inner48 = 163.0 * Rm + 40.0 * (z + y * sqrt2)
    poly48 = 259.0 * R**2 - 6.0 * inner48 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += 2.0 * torch.exp(poly48 / (342.0 * b2))

    # term49: -2 * exp( (487 R^2 - 6(163 Rm + 40(z + y√2))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly49 = 487.0 * R**2 - 6.0 * inner48 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly49 / (342.0 * b2))

    # term50: -1 * exp( (304 R^2 - 120(11 Rm + 2(z + y√2 + x(-√6)))R + 9 Rm(5Rm+16z)) / (342 b2) )
    inner50 = 11.0 * Rm + 2.0 * (z + y * sqrt2 - x * sqrt6)   # because x(-√6)
    poly50 = 304.0 * R**2 - 120.0 * inner50 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -torch.exp(poly50 / (342.0 * b2))

    # term51: -1 * exp( (532 R^2 - 120(11 Rm + 2(z + y√2 + x(-√6)))R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly51 = 532.0 * R**2 - 120.0 * inner50 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -torch.exp(poly51 / (342.0 * b2))

    # term52: -2 * exp( (76 R^2 - 16(18 Rm + 5(z + y√2 + x(-√6)))R + 3 Rm(5Rm+16z)) / (114 b2) )
    inner52 = 18.0 * Rm + 5.0 * (z + y * sqrt2 - x * sqrt6)
    poly52 = 76.0 * R**2 - 16.0 * inner52 * R + 3.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -2.0 * torch.exp(poly52 / (114.0 * b2))

    # term53: -2 * exp( (346 R^2 - 24(49 Rm + 5(z + y√2 + x(-√6)))R + 27 Rm(Rm-8z)) / (342 b2) )
    inner53 = 49.0 * Rm + 5.0 * (z + y * sqrt2 - x * sqrt6)
    poly53 = 346.0 * R**2 - 24.0 * inner53 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -2.0 * torch.exp(poly53 / (342.0 * b2))

    # term54: +2 * exp( (574 R^2 - 24(49 Rm + 5(z + y√2 + x(-√6)))R + 27 Rm(Rm-8z)) / (342 b2) )
    poly54 = 574.0 * R**2 - 24.0 * inner53 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += 2.0 * torch.exp(poly54 / (342.0 * b2))

    # term55: +2 * exp( (422 R^2 - 12(79 Rm + 10(z + y√2 + x(-√6)))R + 27 Rm(Rm-8z)) / (342 b2) )
    inner55 = 79.0 * Rm + 10.0 * (z + y * sqrt2 - x * sqrt6)
    poly55 = 422.0 * R**2 - 12.0 * inner55 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += 2.0 * torch.exp(poly55 / (342.0 * b2))

    # term56: -2 * exp( (650 R^2 - 12(79 Rm + 10(z + y√2 + x(-√6)))R + 27 Rm(Rm-8z)) / (342 b2) )
    poly56 = 650.0 * R**2 - 12.0 * inner55 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -2.0 * torch.exp(poly56 / (342.0 * b2))

    # term57: +4 * exp( (152 R^2 - 12(91 Rm + 20(z + y√2 + x(-√6)))R + 9 Rm(5Rm+16z)) / (342 b2) )
    inner57 = 91.0 * Rm + 20.0 * (z + y * sqrt2 - x * sqrt6)
    poly57 = 152.0 * R**2 - 12.0 * inner57 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 4.0 * torch.exp(poly57 / (342.0 * b2))

    # term58: -2 * exp( (247 R^2 - 6(163 Rm + 40(z + y√2 + x(-√6)))R + 72 Rm(3Rm+2z)) / (342 b2) )
    inner58 = 163.0 * Rm + 40.0 * (z + y * sqrt2 - x * sqrt6)
    poly58 = 247.0 * R**2 - 6.0 * inner58 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly58 / (342.0 * b2))

    # term59: +1 * exp( (475 R^2 - 6(163 Rm + 40(z + y√2 + x(-√6)))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly59 = 475.0 * R**2 - 6.0 * inner58 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += torch.exp(poly59 / (342.0 * b2))

    # term60: +1 * exp( (703 R^2 - 6(163 Rm + 40(z + y√2 + x(-√6)))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly60 = 703.0 * R**2 - 6.0 * inner58 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += torch.exp(poly60 / (342.0 * b2))

    # term61: -1 * exp( (304 R^2 - 120(11 Rm + 2(z + y√2 + x√6))R + 9 Rm(5Rm+16z)) / (342 b2) )
    inner61 = 11.0 * Rm + 2.0 * (z + y * sqrt2 + x * sqrt6)
    poly61 = 304.0 * R**2 - 120.0 * inner61 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -torch.exp(poly61 / (342.0 * b2))

    # term62: -1 * exp( (532 R^2 - 120(11 Rm + 2(z + y√2 + x√6))R + 9 Rm(5Rm+16z)) / (342 b2) )
    poly62 = 532.0 * R**2 - 120.0 * inner61 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -torch.exp(poly62 / (342.0 * b2))

    # term63: -2 * exp( (76 R^2 - 16(18 Rm + 5(z + y√2 + x√6))R + 3 Rm(5Rm+16z)) / (114 b2) )
    inner63 = 18.0 * Rm + 5.0 * (z + y * sqrt2 + x * sqrt6)
    poly63 = 76.0 * R**2 - 16.0 * inner63 * R + 3.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -2.0 * torch.exp(poly63 / (114.0 * b2))

    # term64: -2 * exp( (346 R^2 - 24(49 Rm + 5(z + y√2 + x√6))R + 27 Rm(Rm-8z)) / (342 b2) )
    inner64 = 49.0 * Rm + 5.0 * (z + y * sqrt2 + x * sqrt6)
    poly64 = 346.0 * R**2 - 24.0 * inner64 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -2.0 * torch.exp(poly64 / (342.0 * b2))

    # term65: +2 * exp( (574 R^2 - 24(49 Rm + 5(z + y√2 + x√6))R + 27 Rm(Rm-8z)) / (342 b2) )
    poly65 = 574.0 * R**2 - 24.0 * inner64 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += 2.0 * torch.exp(poly65 / (342.0 * b2))

    # term66: +2 * exp( (422 R^2 - 12(79 Rm + 10(z + y√2 + x√6))R + 27 Rm(Rm-8z)) / (342 b2) )
    inner66 = 79.0 * Rm + 10.0 * (z + y * sqrt2 + x * sqrt6)
    poly66 = 422.0 * R**2 - 12.0 * inner66 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += 2.0 * torch.exp(poly66 / (342.0 * b2))

    # term67: -2 * exp( (650 R^2 - 12(79 Rm + 10(z + y√2 + x√6))R + 27 Rm(Rm-8z)) / (342 b2) )
    poly67 = 650.0 * R**2 - 12.0 * inner66 * R + 27.0 * Rm * (Rm - 8.0 * z)
    bracket_sum += -2.0 * torch.exp(poly67 / (342.0 * b2))

    # term68: +4 * exp( (152 R^2 - 12(91 Rm + 20(z + y√2 + x√6))R + 9 Rm(5Rm+16z)) / (342 b2) )
    inner68 = 91.0 * Rm + 20.0 * (z + y * sqrt2 + x * sqrt6)
    poly68 = 152.0 * R**2 - 12.0 * inner68 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += 4.0 * torch.exp(poly68 / (342.0 * b2))

    # term69: -2 * exp( (247 R^2 - 6(163 Rm + 40(z + y√2 + x√6))R + 72 Rm(3Rm+2z)) / (342 b2) )
    inner69 = 163.0 * Rm + 40.0 * (z + y * sqrt2 + x * sqrt6)
    poly69 = 247.0 * R**2 - 6.0 * inner69 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly69 / (342.0 * b2))

    # term70: +1 * exp( (475 R^2 - 6(163 Rm + 40(z + y√2 + x√6))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly70 = 475.0 * R**2 - 6.0 * inner69 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += torch.exp(poly70 / (342.0 * b2))

    # term71: +1 * exp( (703 R^2 - 6(163 Rm + 40(z + y√2 + x√6))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly71 = 703.0 * R**2 - 6.0 * inner69 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += torch.exp(poly71 / (342.0 * b2))

    # term72: -4 * exp( (164 R^2 - 12(91 Rm + 10(2z - √2 y + x√6))R + 9 Rm(5Rm+16z)) / (342 b2) )
    inner72 = 91.0 * Rm + 10.0 * (2.0 * z - sqrt2 * y + x * sqrt6)
    poly72 = 164.0 * R**2 - 12.0 * inner72 * R + 9.0 * Rm * (5.0 * Rm + 16.0 * z)
    bracket_sum += -4.0 * torch.exp(poly72 / (342.0 * b2))

    # term73: +2 * exp( (259 R^2 - 6(163 Rm + 20(2z - √2 y + x√6))R + 72 Rm(3Rm+2z)) / (342 b2) )
    inner73 = 163.0 * Rm + 20.0 * (2.0 * z - sqrt2 * y + x * sqrt6)
    poly73 = 259.0 * R**2 - 6.0 * inner73 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += 2.0 * torch.exp(poly73 / (342.0 * b2))

    # term74: -2 * exp( (487 R^2 - 6(163 Rm + 20(2z - √2 y + x√6))R + 72 Rm(3Rm+2z)) / (342 b2) )
    poly74 = 487.0 * R**2 - 6.0 * inner73 * R + 72.0 * Rm * (3.0 * Rm + 2.0 * z)
    bracket_sum += -2.0 * torch.exp(poly74 / (342.0 * b2))

    # ---------- denominator sum (inside the big denominator) ----------
    denom_sum = (-10.0 * torch.exp(4.0 * R**2 / (9.0 * b2))
                 - 5.0 * torch.exp(10.0 * R**2 / (9.0 * b2))
                 + 24.0 * torch.exp(2.0 * R * Rm / (3.0 * b2))
                 - 6.0 * torch.exp(2.0 * R * (R + 2.0 * Rm) / (3.0 * b2))
                 + 6.0 * torch.exp(2.0 * R * (2.0 * R + 3.0 * Rm) / (9.0 * b2))
                 - 9.0 * torch.exp(2.0 * R * (R + 6.0 * Rm) / (9.0 * b2))
                 - 15.0 * torch.exp((5.0 * R**2 + 18.0 * Rm * R + 9.0 * Rm**2) / (18.0 * b2))
                 + 10.0 * torch.exp((17.0 * R**2 + 18.0 * Rm * R + 9.0 * Rm**2) / (18.0 * b2))
                 + 5.0 * torch.exp((29.0 * R**2 + 18.0 * Rm * R + 9.0 * Rm**2) / (18.0 * b2)))

    # ---------- full denominator ----------
    denominator = (b ** 3) * (-1.0 + torch.exp(2.0 * R**2 / (3.0 * b2))) * denom_sum

    # ---------- final result ----------
    #result = const1 * num_factor * bracket_sum / denominator
    logf = log_const1+torch.log(num_factor)+torch.log(bracket_sum) - torch.log(denominator)
    logf = torch.clamp(logf, min=-45, max=45)
    f = torch.exp(logf)

    return f * vol, logf   

def fden_3D(x: torch.Tensor, param: dict):
    device, dtype = x.device, x.dtype
    Nd = param['nd']

    # ---- domain transform ----
    a = torch.tensor(param['xmin'], dtype=dtype, device=device)[:Nd]
    b = torch.tensor(param['xmax'], dtype=dtype, device=device)[:Nd]
    x_phys = (b - a) * x + a
    vol = torch.prod(b - a)

    # ---- parameters ----
    b_  = torch.tensor(param['b'], dtype=dtype, device=device)
    R_  = torch.tensor(param['R'], dtype=dtype, device=device)
    Rm_ = torch.tensor(param['Rm'], dtype=dtype, device=device)

    x0, x1, x2 = x_phys[..., 0], x_phys[..., 1], x_phys[..., 2]

    sqrt2 = math.sqrt(2.0)
    sqrt6 = math.sqrt(6.0)

    b2 = b_**2

    # =========================================================
    # 1. GLOBAL GAUSSIAN FACTOR (log form)
    # =========================================================
    log_pref = (
        math.log(40.0) + 0.5 * math.log(5/19)
        - (1420*R_**2 - 6840*Rm_*R_ + 297*Rm_**2
           + 1800*(x0**2 + x1**2 + x2**2)) / (1710*b2)
    )

    # =========================================================
    # 2. BUILD EXPONENTIAL SUM USING LOG-SUM-EXP
    # =========================================================

    terms = []

    def add_term(coeff, num, denom):
        log_term = torch.log(torch.abs(torch.tensor(coeff, dtype=dtype, device=device))) \
                   + num / denom
        sign = torch.sign(torch.tensor(coeff, dtype=dtype, device=device))
        terms.append((log_term, sign))

    # ---- reusable structures ----
    mix1 = (-2*x2 + x1*sqrt2 + x0*sqrt6)
    mix2 = (2*x2 - sqrt2*x1 + sqrt6*x0)
    mix3 = (x2 + sqrt2*x1)
    mix4 = (x2 - 2*sqrt2*x1)

    # ---- representative full structure ----
    add_term(-6, 194*R_**2 - 24*(43*Rm_ - 15*x2)*R_ + 27*Rm_*(Rm_ - 8*x2), 342*b2)
    add_term(+6, 422*R_**2 - 24*(43*Rm_ - 15*x2)*R_ + 27*Rm_*(Rm_ - 8*x2), 342*b2)

    add_term(+4, 346*R_**2 - 180*(7*Rm_ - 2*x2)*R_ + 27*Rm_*(Rm_ - 8*x2), 342*b2)
    add_term(-2, 574*R_**2 - 180*(7*Rm_ - 2*x2)*R_ + 27*Rm_*(Rm_ - 8*x2), 342*b2)

    add_term(+2, 259*R_**2 - 6*(179*Rm_ + 20*mix1)*R_ + 72*Rm_*(3*Rm_ + 2*x2), 342*b2)
    add_term(-2, 487*R_**2 - 6*(179*Rm_ + 20*mix1)*R_ + 72*Rm_*(3*Rm_ + 2*x2), 342*b2)

    add_term(+2, 259*R_**2 - 1074*Rm_*R_ + 240*mix3*R_ + 72*Rm_*(3*Rm_ + 2*x2), 342*b2)
    add_term(-2, 487*R_**2 - 1074*Rm_*R_ + 240*mix3*R_ + 72*Rm_*(3*Rm_ + 2*x2), 342*b2)

    add_term(+2, 259*R_**2 - 1074*Rm_*R_ + 120*mix2*R_ + 72*Rm_*(3*Rm_ + 2*x2), 342*b2)
    add_term(-2, 487*R_**2 - 1074*Rm_*R_ + 120*mix2*R_ + 72*Rm_*(3*Rm_ + 2*x2), 342*b2)

    # =========================================================
    # 3. STABLE SUM (separate + and -)
    # =========================================================
    log_pos = []
    log_neg = []

    for log_term, sign in terms:
        if sign > 0:
            log_pos.append(log_term)
        else:
            log_neg.append(log_term)

    def logsumexp_list(log_list):
        if len(log_list) == 0:
            return torch.tensor(-1e30, dtype=dtype, device=device)
        stacked = torch.stack(log_list, dim=0)
        m = torch.max(stacked, dim=0).values
        return m + torch.log(torch.sum(torch.exp(stacked - m), dim=0))

    log_S_pos = logsumexp_list(log_pos)
    log_S_neg = logsumexp_list(log_neg)

    # signed log difference
    S = torch.exp(log_S_pos) - torch.exp(log_S_neg)
    log_S = torch.log(torch.abs(S) + 1e-30)

    # =========================================================
    # 4. DENOMINATOR (log form)
    # =========================================================
    log_denom = (
        math.log(19.0) + 3*torch.log(b_)
        + torch.log(torch.abs(-1 + torch.exp(2*R_**2/(3*b2))) + 1e-30)
        + 1.5*math.log(math.pi)
    )

    # (dominant exponential term grouping)
    denom_terms = torch.stack([
        4*R_**2/(9*b2),
        10*R_**2/(9*b2),
        2*R_*Rm_/(3*b2),
        2*R_*(R_+2*Rm_)/(3*b2)
    ], dim=0)

    log_denom += torch.logsumexp(denom_terms, dim=0)

    # =========================================================
    # 5. FINAL RESULT
    # =========================================================
    logf = log_pref + log_S - log_denom

    logf = torch.clamp(logf, min=-45, max=45)
    f = torch.exp(logf)

    return f * vol, logf
 
 

 

 

def rho_ws_all(X, R0, w, a, beta2, gamma, beta3, beta4):
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)

    theta = torch.acos(z / (r + 1e-12))

    # 修改这里：使用 torch.ones_like 或 torch.tensor 并指定 device
    pi = torch.acos(torch.tensor(-1.0, device=X.device, dtype=X.dtype))

    # 修改这些常数的创建方式
    Y20 = 0.25 * torch.sqrt(torch.tensor(5.0, device=X.device, dtype=X.dtype)/pi) * (3*torch.cos(theta)**2 - 1)
    Y30 = 0.25 * torch.sqrt(torch.tensor(7.0, device=X.device, dtype=X.dtype)/pi) * (5*torch.cos(theta)**3 - 3*torch.cos(theta))
    Y40 = (3/(16*torch.sqrt(pi))) * (35*torch.cos(theta)**4 - 30*torch.cos(theta)**2 + 3)

    R = R0 * (1 + beta2*Y20 + beta3*Y30 + beta4*Y40)

    # 修改这里的常数
    Val = torch.tensor(0.000817472, device=X.device, dtype=X.dtype) * (1 + w * (r**2 / R0**2)) \
          / (1 + torch.exp((r - R) / a)) \
          * (4 * pi)

    return Val




def calculate_jastrow_factor(X, a, b, c, d, e, f):
    B, N, _ = X.shape

    diff = X[:, :, None, :] - X[:, None, :, :]
    rij = torch.linalg.norm(diff, dim=-1)

    term = 1 + a * torch.exp(-b * rij**2) * (
        c + d * rij**2 + e * rij**3 + f * rij**4
    )

    eps = 1e-12
    term = torch.clamp(term, min=eps)

    # Get upper triangle indices once
    i, j = torch.triu_indices(N, N, offset=1, device=X.device)

    log_J = torch.sum(torch.log(term[:, i, j]), dim=1)
    J = torch.exp(log_J)

    return J, log_J

def calculate_jastrow_factor_default2(X, a, b, c, d, e, f):
    B, N, _ = X.shape

    # Pairwise distances
    diff = X[:, :, None, :] - X[:, None, :, :]
    rij = torch.linalg.norm(diff, dim=-1)

    # Upper triangular mask (i < j)
    mask = torch.triu(
        torch.ones(N, N, device=X.device, dtype=X.dtype),
        diagonal=1
    )

    # Jastrow term
    term = 1 + a * torch.exp(-b * rij**2) * (
        c + d * rij**2 + e * rij**3 + f * rij**4
    )

    # Numerical safety (optional but recommended)
    eps = 1e-12
    term = torch.clamp(term, min=eps)

    # Compute log only for i < j
    log_term = torch.log(term) * mask

    # Sum over particle pairs
    log_J = torch.sum(log_term, dim=(1, 2))

    # Exponentiate
    J = torch.exp(log_J)

    return J, log_J

def calculate_jastrow_factor_default(X, a, b, c, d, e, f):
    B, N, _ = X.shape

    diff = X[:, :, None, :] - X[:, None, :, :]
    rij = torch.linalg.norm(diff, dim=-1)

    mask = torch.triu(
        torch.ones(N, N, device=X.device, dtype=X.dtype),
        diagonal=1
    )

    term = 1 + a * torch.exp(-b * rij**2) * (
        c + d*rij**2 + e*rij**3 + f*rij**4
    )

    term = term * mask + (1 - mask)

    J = torch.prod(torch.prod(term, dim=2), dim=1)
    return J

def calculate_jastrow_factor_streaming(X, a, b, c, d, e, f):
    """
    X: (B, N, 3)
    Computes Jastrow without forming full (B,N,N) tensor
    """
    B, N, _ = X.shape
    #J = torch.ones(B, device=X.device, dtype=X.dtype)
    log_J = torch.zeros(B, device=X.device, dtype=X.dtype)

    for i in range(N):
        xi = X[:, i:i+1, :]           # (B,1,3)
        rij = torch.linalg.norm(X[:, i+1:, :] - xi, dim=2)  # (B, N-i-1)

        term = 1 + a * torch.exp(-b * rij**2) * (
            c + d*rij**2 + e*rij**3 + f*rij**4
        )
        log_J = log_J+torch.sum(torch.log(term),dim=1)

        #J = J * torch.prod(term, dim=1)

    J = torch.exp(log_J)
    return J, log_J


def fun_ws(x: torch.Tensor, param: dict):
    device, dtype = x.device, x.dtype
    nd = param['nd']
    Isrc = param['Isrc']
    Inucleus = param['Inucleus']

    # 修改这里：确保这些张量连接到计算图
    a = torch.tensor(param['xmin'][:nd], device=device, dtype=dtype, requires_grad=False)
    b = torch.tensor(param['xmax'][:nd], device=device, dtype=dtype, requires_grad=False)
    scl = param['scale']

    if x.ndim == 1:
        x = x.view(1, nd)
    elif x.shape[1] != nd:
        x = x.t()

    x = (b - a) * x + a #x_phys

    B = x.shape[0]
    nParticles = nd // 3
    X = x.view(B, nParticles, 3)

    # ===== nucleus parameters =====
    # 修改所有常数的创建方式
    if Inucleus == 208:
        R0 = torch.tensor(6.49, device=device, dtype=dtype, requires_grad=False)
        w  = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=False)
        aa = torch.tensor(0.54, device=device, dtype=dtype, requires_grad=False)
        beta2 = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=False)
        beta3 = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=False)
        beta4 = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=False)
        gamma = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=False)
        Nor = torch.tensor(2.4932e4, device=device, dtype=dtype, requires_grad=False)

    elif Inucleus == 197:
        R0 = torch.tensor(6.56, device=device, dtype=dtype, requires_grad=False)
        w  = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=False)
        aa = torch.tensor(0.48, device=device, dtype=dtype, requires_grad=False)
        beta2 = torch.tensor(0.135, device=device, dtype=dtype, requires_grad=False)
        beta3 = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=False)
        beta4 = torch.tensor(-0.023, device=device, dtype=dtype, requires_grad=False)
        gamma = torch.tensor(43/90*3.14159/2.0, device=device, dtype=dtype, requires_grad=False)
        Nor = torch.tensor(2.6657e4, device=device, dtype=dtype, requires_grad=False)

    # volume
    log_vol = torch.sum(torch.log(b - a))

    # Woods–Saxon
    mat = rho_ws_all(X, R0, w, aa, beta2, gamma, beta3, beta4)
    log_val = torch.sum(torch.log(mat), dim=1)

    # Jastrow
    if Isrc == 1:
        if nd <193:
            J, log_J = calculate_jastrow_factor(X, -1,1.2627,0.9916,2.1032,-6.1297,2.9348)
        else:
            J, log_J = calculate_jastrow_factor_streaming(X, -1,1.2627,0.9916,2.1032,-6.1297,2.9348)
    elif Isrc == 2:
        J, log_J = calculate_jastrow_factor_streaming(X, -1,1.2746,0.3820,-0.8557,0.1328,0.0849)
    elif Isrc == 3:
        J, log_J = calculate_jastrow_factor_streaming(X, -1,1.9319,0.9868,2.5145,-4.7581,0.0555)
    elif Isrc == 4:
        J, log_J = calculate_jastrow_factor_streaming(X, -1,1.3977,0.9965,1.8195,-1.6541,-0.3312)
    else:
        J = torch.ones(B, device=device, dtype=dtype)
        log_J = torch.log(J)

    # 修改这里的常数创建
    log_scl = torch.log(torch.tensor(scl, device=device, dtype=dtype, requires_grad=False))

    log_f = log_J + log_val + log_vol + nd*log_scl - torch.log(Nor)*(nd/12.0)
    
    log_f = torch.clamp(log_f, min=-0.4*nd, max=1.5*nd)
    f = torch.exp(log_f)

    return f, log_f




def fun_2D(x: torch.Tensor, param: dict) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    Nd = param['nd']

    a = to_tensor(param['xmin'], dtype, device)[:Nd]
    b = to_tensor(param['xmax'], dtype, device)[:Nd]

    x_phys = (b - a) * x + a
    vol = torch.prod(b - a)

    RA = to_tensor(param['Ra'], dtype, device)
    RB = to_tensor(param['Rb'], dtype, device)
    bA = to_tensor(param['b1'], dtype, device)
    bB = to_tensor(param['b2'], dtype, device)

    def phi(r, R, bb):
        return torch.exp(-(r - R)**2 / (2 * bb**2)) / (math.pi * bb**2)**0.25

    r0, r1 = x_phys[..., 0], x_phys[..., 1]

    phi_2p = (phi(r0, RA, bA) * phi(r1, RB, bB) -
              phi(r0, RB, bB) * phi(r1, RA, bA)) / math.sqrt(2)

    S = torch.sqrt(2*bA*bB/(bA**2+bB**2)) * \
        torch.exp(-(RA-RB)**2/(2*(bA**2+bB**2)))

    f = (phi_2p**2)/(1 - S**2)* vol
    
    logf = torch.log(f)
    
    logf = torch.clamp(logf, min=-45, max=45)

    f = torch.exp(logf)  #.unsqueeze(-1)  # (B,1)
    return f,logf  

# custom function
#def _cluster_centers(n_particles: int, l: float, device=None, dtype=torch.float64):
#    """
#    Return tensor of shape (n_particles, 3) containing cluster centers for indices 1..n_particles.
#   Supports up to 4 clusters (matching your MATLAB Checkcluster cases).
#    """
#    if n_particles > 4:
#        raise ValueError("Checkcluster in original Matlab only supports up to 4 clusters.")
#    centers = torch.zeros((n_particles, 3), dtype=dtype, device=device)
#    # case 1
    
#    centers[0] = torch.tensor([0.0, 2*math.sqrt(2)/3*l, -l/3], dtype=dtype, device=device)
#    
#    if n_particles >= 2:
#        centers[1] = torch.tensor([-math.sqrt(6)/3*l, -math.sqrt(2)/3*l, -l/3], dtype=dtype, device=device)
        
#    if n_particles >= 3:
#        centers[2] = torch.tensor([0.0, 0.0, l], dtype=dtype, device=device)
        
#    if n_particles >= 4:
 #       centers[3] = torch.tensor([math.sqrt(6)/3*l, -math.sqrt(2)/3*l, -l/3], dtype=dtype, device=device)
 #   return centers

def _cluster_centers(n_particles: int, l: float, device=None, dtype=torch.float64):
    """
    Return tensor of shape (n_particles, 3) containing cluster centers for indices 1..n_particles.
    Supports up to 4 clusters (matching your MATLAB Checkcluster cases).
    """
    #if n_particles > 4:
    #    raise ValueError("Checkcluster in original Matlab only supports up to 4 clusters.")
    centers = torch.zeros((n_particles, 3), dtype=dtype, device=device)
    # case 1
    
    centers[0] = torch.tensor([0.0, 2*math.sqrt(2)/3*l, -l/3], dtype=dtype, device=device)
    
    if n_particles == 2:
        centers[1] = torch.tensor([-math.sqrt(6)/3*l, -math.sqrt(2)/3*l, -l/3], dtype=dtype, device=device)
        
    elif n_particles == 3:
        centers[1] = torch.tensor([-math.sqrt(6)/3*l, -math.sqrt(2)/3*l, -l/3], dtype=dtype, device=device)
        centers[2] = torch.tensor([0.0, 0.0, l], dtype=dtype, device=device)
        
    elif n_particles == 4:
        centers[1] = torch.tensor([-math.sqrt(6)/3*l, -math.sqrt(2)/3*l, -l/3], dtype=dtype, device=device)
        centers[2] = torch.tensor([0.0, 0.0, l], dtype=dtype, device=device)        
        centers[3] = torch.tensor([math.sqrt(6)/3*l, -math.sqrt(2)/3*l, -l/3], dtype=dtype, device=device)

    elif n_particles == 6:
        # Regular octahedron: (±1,0,0), (0,±1,0), (0,0,±1)
        centers = torch.tensor([
            [ l, 0, 0],
            [-l, 0, 0],
            [ 0, l, 0],
            [ 0,-l, 0],
            [ 0, 0, l],
            [ 0, 0,-l],
        ], dtype=dtype, device=device)

    elif n_particles == 8:
        # Cube: all sign combinations of (±1,±1,±1)
        centers = torch.tensor([
            [ l,  l,  l],
            [ l,  l, -l],
            [ l, -l,  l],
            [ l, -l, -l],
            [-l,  l,  l],
            [-l,  l, -l],
            [-l, -l,  l],
            [-l, -l, -l],
        ], dtype=dtype, device=device)

    elif n_particles == 12:
        # Icosahedron
        phi = (1 + math.sqrt(5)) / 2*l  # golden ratio
        centers = torch.tensor([
            [ 0,  l,  phi],
            [ 0, -l,  phi],
            [ 0,  l, -phi],
            [ 0, -l, -phi],
            [ l,  phi, 0],
            [-l,  phi, 0],
            [ l, -phi, 0],
            [-l, -phi, 0],
            [ phi, 0,  l],
            [-phi, 0,  l],
            [ phi, 0, -l],
            [-phi, 0, -l],
        ], dtype=dtype, device=device)

    elif n_particles == 20:
        # Dodecahedron
        phi = (1 + math.sqrt(5)) / 2*l
        a = l*l / phi

        centers = torch.tensor([
            # (±1, ±1, ±1)
            [ l,  l,  l],
            [ l,  l, -l],
            [ l, -l,  l],
            [ l, -l, -l],
            [-l,  l,  l],
            [-l,  l, -l],
            [-l, -l,  l],
            [-1, -1, -1],

            # (0, ±a, ±phi)
            [ 0,  a,  phi],
            [ 0,  a, -phi],
            [ 0, -a,  phi],
            [ 0, -a, -phi],

            # (±a, ±phi, 0)
            [  a,  phi, 0],
            [  a, -phi, 0],
            [ -a,  phi, 0],
            [ -a, -phi, 0],

            # (±phi, 0, ±a)
            [  phi, 0,  a],
            [  phi, 0, -a],
            [ -phi, 0,  a],
            [ -phi, 0, -a],
        ], dtype=dtype, device=device)

    else:
        raise ValueError("Supported n_particles: 2, 3, 4, 6, 8, 12, 20")

    # Normalize so average radius = l (nice physical scaling)
    #radii = torch.norm(centers, dim=1, keepdim=True)
    #centers = centers / radii.mean() * l

    return centers


 

 





def fun_cluster(x: torch.Tensor, param: dict) -> torch.Tensor:
#def fun_cluster_torch(x: torch.Tensor, param: dict) -> torch.Tensor:
    """
    Stable PyTorch version of the Matlab fun_cluster with robust log-det computation.
    Returns f with shape (n_samples, 1).
    """
    device, dtype = x.device, x.dtype

    nd = int(param['nd'])
    scl = param['scale']
    a = _to_tensor(param['xmin'], dtype=dtype, device=device)[:nd]
    b = _to_tensor(param['xmax'], dtype=dtype, device=device)[:nd]

    vol = torch.prod(b - a)

    # map x from [0,1] to [a,b]
    x = (b - a) * x + a  # shape (n_samples, nd)

    n_samples, n2 = x.shape
    assert n2 == nd, "x second dimension must equal nd"

    nParticles = nd // 3
    if nd % 3 != 0:
        raise ValueError("nd must be divisible by 3")

    # constants
    l = 3.0
    bw = 1.3
    nu = 1.0 / (2.0 * (bw ** 2))
    Nor = 1.0 - 3.0 * math.exp(-8.0 * l**2 / (3.0 * bw**2)) \
          + 8.0 * math.exp(-2.0 * l**2 / (bw**2)) \
          - 6.0 * math.exp(-4.0 * l**2 / (3.0 * bw**2))

    coords = x.view(n_samples, nParticles, 3)

    # cluster centers
    centers = _cluster_centers(nParticles, l, device=device, dtype=dtype)  # (P,3)

    C = (2.0 * nu / math.pi) ** (3.0 / 4.0)

    # pairwise squared distances
    coords_exp = coords.unsqueeze(2)                  # (B,P,1,3)
    centers_exp = centers.unsqueeze(0).unsqueeze(0)   # (1,1,P,3)
    D2 = ((coords_exp - centers_exp) ** 2).sum(dim=-1)  # (B,P,P)

    mat = C * torch.exp(-nu * D2)         # (B,P,P)

    # Gram matrix (symmetric)
    Gram = torch.matmul(mat.transpose(1, 2), mat)
    Gram = 0.5 * (Gram + Gram.transpose(1, 2))  # enforce symmetry

    # compute a per-batch jitter scaled to Gram magnitude (helps small & large scales)
    # trace_sum shape (B,)
    trace_vals = torch.diagonal(Gram, dim1=-2, dim2=-1).sum(-1)
    # baseline eps scaled by average diagonal magnitude (avoid zero)
    eps_base = 1e-8
    eps_scale = 1e-6
    eps_batch = eps_base + eps_scale * torch.clamp(trace_vals / (Gram.shape[-1] + 1.0), min=0.0)
    # ensure eps_batch is at least eps_base and shaped (B,)
    # we'll try increasing jitter until cholesky succeeds (loop over small number of retries)
    B, P, _ = Gram.shape
    I = torch.eye(P, dtype=dtype, device=device).expand(B, P, P)  # (B,P,P)

    # Try Cholesky with adaptive jitter
    max_tries = 6
    jitter = eps_batch.clone()  # (B,)
    success_mask = torch.zeros(B, dtype=torch.bool, device=device)
    logabsdet = torch.empty(B, dtype=dtype, device=device)
    # initialize with -inf
    logabsdet[:] = float("-inf")

    # We'll perform up to max_tries attempts; collect which batches succeed
    Gram_work = Gram.clone()
    for attempt in range(max_tries):
        # form Gram + jitter * I (broadcast)
        jitter_mat = jitter.view(B, 1, 1) * I  # (B,P,P)
        Gram_try = Gram_work + jitter_mat

        # symmetrize again for safety
        Gram_try = 0.5 * (Gram_try + Gram_try.transpose(1, 2))

        # use cholesky_ex which returns (L, info) and doesn't throw on GPU
        L, info = torch.linalg.cholesky_ex(Gram_try)
        #L=torch.linalg.cholesky(Gram_try)
       

        # info == 0 indicates success for that batch element
        this_success = (info == 0)
        newly_succeed = this_success & (~success_mask)

        if newly_succeed.any():
            # compute logdet for newly succeeded batches
            L_succ = L[newly_succeed]  # (k,P,P)
            diag = torch.diagonal(L_succ, dim1=-2, dim2=-1)  # (k,P)
            logdet_succ = 2.0 * torch.log(torch.clamp(diag, min=1e-12)).sum(dim=-1)  # (k,)
            logabsdet[newly_succeed] = logdet_succ
            success_mask = success_mask | newly_succeed

        if success_mask.all():
            break

        # for failed batches increase jitter multiplicatively and retry
        jitter = torch.where(this_success, jitter, jitter * 10.0)

    # After tries, for any remaining failed batches, fallback to stable slogdet but with large jitter
    if (~success_mask).any():
        idx_fail = (~success_mask).nonzero(as_tuple=False).squeeze(-1)
        # add larger jitter for failed ones
        large_jitter = (jitter[idx_fail].clamp_min(1e-4)).view(-1, 1, 1) * torch.eye(P, dtype=dtype, device=device)
        Gram_fallback = Gram[idx_fail] + large_jitter
        # symmetrize
        Gram_fallback = 0.5 * (Gram_fallback + Gram_fallback.transpose(-1, -2))
        # use slogdet fallback (should be stable now)
        sign_fb, logabs_fb = torch.linalg.slogdet(Gram_fallback)
        # ensure finite values
        logabs_fb = torch.where(torch.isfinite(logabs_fb), logabs_fb, torch.full_like(logabs_fb, -1e6))
        # place into logabsdet
        logabsdet[idx_fail] = logabs_fb

    # At this point logabsdet should be finite for all batches (but clamp just in case)
    logabsdet = torch.clamp(logabsdet, min=-50, max=50)

    # Build prefactor and final logf
    log_fact = math.lgamma(nParticles + 1)
    log_prefactor = -log_fact - math.log(Nor) + math.log(vol)
    
    log_prefactor = log_prefactor+nd*math.log(scl)  # additional factor scl^nd

    logf = logabsdet + log_prefactor
    # kill any negative sign situations by setting very small value (we already used cholesky so sign>0)
    logf = torch.clamp(logf, min=-45, max=45)

    f = torch.exp(logf)  #.unsqueeze(-1)  # (B,1)

    return f, logf


class CustomDistribution():
    def __init__(self,fun_custom, param):
        #super().__init__(validate_args=validate_args) 
        self.fun = fun_custom        
        self.param= param  

    def log_prob(self, x):
        # x: [B, nd] tensor
        device = x.device
        dtype = x.dtype
        nd = int(self.param['nd'])

        # map xmin/xmax to tensors
        a = torch.tensor(self.param['xmin'], device=device, dtype=dtype)[:nd]
        b = torch.tensor(self.param['xmax'], device=device, dtype=dtype)[:nd]

        #vol = torch.prod(b - a)
        log_vol = torch.log(b-a)
        log_vol=log_vol.sum()

        # map x from [a,b] to [0,1]
        x_scaled = (x - a) / (b - a)

        # compute target density
        f,log_f = self.fun(x_scaled,self.param)
        #f = torch.clamp(f, min=1e-40, max=1e40)  # prevent log(0)
        
        log_f = log_f-log_vol
        # log-probability
        #return torch.log(f) - torch.log(vol)  # shape: [B]  
        return log_f