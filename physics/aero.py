def downforce(rho: float, CL_A: float, v: float) -> float:
    return 0.5 * rho * CL_A * v*v

def drag(rho: float, CD_A: float, v: float) -> float:
    return 0.5 * rho * CD_A * v*v