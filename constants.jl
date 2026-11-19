const SOLVER_ED   = "ed"
const SOLVER_DMRG = "dmrg"
const SOLVER_IDMRG = "idmrg"
const SOLVERS = (SOLVER_ED, SOLVER_DMRG, SOLVER_IDMRG)


const MODEL_RABI = "rabi"
const MODEL_XXZ  = "xxz"
const MODEL_TFIM = "tfim"
const MODELS = (MODEL_RABI, MODEL_XXZ, MODEL_TFIM)


const COUPLING_RBH    = "rbh"
const COUPLING_NATIVE = "native"
const COUPLING_SOURCES = (COUPLING_RBH, COUPLING_NATIVE)

const PARAM_LATTICE     = "lattice"
const PARAM_MICROSCOPIC = "microscopic"
const PARAMETRIZATIONS = (PARAM_LATTICE, PARAM_MICROSCOPIC)

const OBS_ENERGY = :energy
const OBS_ENERGY_PER_PARTICLE = :energy_per_particle
const OBS_P = :polarization
const OBS_P2 = :square_polarization
const OBS_P4 = :quartic_polarization
const OBS_PX = :transverse_polarization
const OBS_POLARIZATION_CORRELATIONS = :polarization_correlation_matrix
const OBS_SPIN_SPIN_CORRELATIONS = :spin_spin_correlations
# OBS_SK uses k=0; the spectrum includes every discrete momentum.
const OBS_SK = :structure_factor
const OBS_SK_SPECTRUM = :structure_factor_spectrum
const AVAILABLE_OBSERVABLES = (
    OBS_ENERGY,
    OBS_ENERGY_PER_PARTICLE,
    OBS_P,
    OBS_P2,
    OBS_P4,
    OBS_PX,
    OBS_POLARIZATION_CORRELATIONS,
    OBS_SPIN_SPIN_CORRELATIONS,
    OBS_SK,
    OBS_SK_SPECTRUM,
)
