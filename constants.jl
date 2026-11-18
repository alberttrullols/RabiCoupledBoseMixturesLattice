const SOLVER_ED   = "ed"
const SOLVER_DMRG = "dmrg"
const SOLVERS = (SOLVER_ED, SOLVER_DMRG)

# Which physical model's Hamiltonian is being solved.
#   - MODEL_RABI: full two-component Rabi-coupled Bose-Hubbard ring.
#   - MODEL_XXZ:  effective spin-1/2 XXZ ring in transverse+longitudinal field.
#   - MODEL_TFIM: the same, with the sigma+sigma- ("XY") term switched off.
const MODEL_RABI = "rabi"
const MODEL_XXZ  = "xxz"
const MODEL_TFIM = "tfim"
const MODELS = (MODEL_RABI, MODEL_XXZ, MODEL_TFIM)

# How a spin model (MODEL_XXZ/MODEL_TFIM) is parametrized; irrelevant for
# MODEL_RABI.
#   - COUPLING_RBH: derive the spin couplings from Rabi-Bose-Hubbard
#     parameters (J, U, U_ab, t, h) via 2nd-order perturbation theory, so
#     results are directly comparable to the full Rabi-BH model.
#   - COUPLING_NATIVE: give the spin couplings (Jxy, Jz, hx, hz) directly.
const COUPLING_RBH    = "rbh"
const COUPLING_NATIVE = "native"
const COUPLING_SOURCES = (COUPLING_RBH, COUPLING_NATIVE)

const PARAM_LATTICE     = "lattice"
const PARAM_MICROSCOPIC = "microscopic"
const PARAMETRIZATIONS = (PARAM_LATTICE, PARAM_MICROSCOPIC)
