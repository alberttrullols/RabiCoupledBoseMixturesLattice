include(joinpath(@__DIR__, "generate_folders.jl"))

const MODEL = MODEL_RABI                # or MODEL_XXZ / MODEL_TFIM
const SOLVER = SOLVER_ED
const PARAMETRIZATION = PARAM_LATTICE    # or PARAM_MICROSCOPIC
const COUPLING_SOURCE = COUPLING_RBH     # only used when MODEL != MODEL_RABI; or COUPLING_NATIVE
const PREFIX = "rabi_ring"
 
size_pairs = [(2, 2)]

t_values = [0.05 * 2.0353774728894862] # to keep in mind here that Omega = 2t (for L.Barbiero et al., Zhan et al.)
h_values = [0.0]

N1_values = Int[]  
const DMC_SHIFT = false  # shift energies to the continuum "DMC zero" via mathieu_band_edges

# DMRG/iDMRG solver settings
nsweeps = 10
maxdims = [10, 20, 50, 100, 200]
cutoff = 1e-10
unitcell_size = 2
max_occupancy = 2
nsteps_per_stage = 3

parent_folder = PREFIX

# BH Lattice parameters (U = U_a = U_b assumed). 
J_values    = [0.0946616187197446]
U_values    = [2.0353774728894862]
U_ab_values = [2 * 2.0353774728894862]

# Microscopic contact interactions (g = g_up = g_down assumed). 
g_values   = Float64[]
g12_values = Float64[]

# Lattice depth in recoil units (V0 / E_r). 
V0_values = [10.0]
kL = Float64(π)  

# Spin-native ring sizes. 
L_values = Int[]

# Native spin couplings 
Jxy_values = Float64[]
Jz_values  = Float64[]
hx_values  = Float64[]
hz_values  = Float64[]

total = generate_folders(;
    model=MODEL, solver=SOLVER, parametrization=PARAMETRIZATION, coupling_source=COUPLING_SOURCE,
    size_pairs, L_values,
    J_values, U_values, U_ab_values, g_values, g12_values,
    Jxy_values, Jz_values, hx_values, hz_values,
    t_values, h_values, V0_values, kL, N1_values, dmc_shift=DMC_SHIFT,
    prefix=PREFIX, parent_folder, nsweeps, maxdims, cutoff,
    unitcell_size, max_occupancy, nsteps_per_stage,
)

println("Total folders created: $total")
