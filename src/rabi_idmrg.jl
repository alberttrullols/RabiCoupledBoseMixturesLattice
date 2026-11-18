using ITensors
using ITensorMPS
import iDMRG

isdefined(@__MODULE__, :RabiModel) || include(joinpath(@__DIR__, "models", "rabi_model.jl"))
using .RabiModel

mutable struct RabiIDMRGEnergyObserver
    energy_density::Union{Nothing,Float64}
end

function iDMRG.checkdone!(observer::RabiIDMRGEnergyObserver; eng_density, kwargs...)
    observer.energy_density = Float64(eng_density)
    return false
end

function _rabi_idmrg_hamiltonians(params::RabiParams, unitcell_size::Int, max_occupancy::Int)
    nphysical = 6 * unitcell_size
    finite_params = RabiParams(;
        N=nphysical, L=nphysical, J=params.J, U_a=params.U_a, U_b=params.U_b,
        U_ab=params.U_ab, t=params.t, h=params.h, V0=params.V0, kL=params.kL,
    )
    finite_sites = siteinds(
        "Boson", 2nphysical; dim=max_occupancy + 1, conserve_qns=true,
    )
    finite_hamiltonian =
        RabiModel.build_mpo_hamiltonian(finite_params, finite_sites; boundary=:open)

    nsites_per_cell = 2 * unitcell_size
    cell_sites = siteinds(
        "Boson", nsites_per_cell; dim=max_occupancy + 1, conserve_qns=true,
    )
    bulk_hamiltonian = iDMRG.initializeHam(
        finite_hamiltonian, nsites_per_cell, cell_sites;
        site_start=3 * nsites_per_cell - 1, ham_uc=2,
    )
    return finite_hamiltonian, bulk_hamiltonian, cell_sites
end


function two_component_idmrg_ground_state(
    params::RabiParams;
    unitcell_size::Int=2,
    max_occupancy::Int=2,
    dmc_shift::Bool=false,
    nsweeps::Int=10,
    maxdims::Vector{Int}=[10, 20, 50, 100, 200],
    cutoff::Real=1e-10,
    nsteps_per_stage::Int=3,
    eigsolve_krylovdim::Int=3,
    init_state::Union{Nothing,Vector{String}}=nothing,
)
    params.N == params.L ||
        throw(ArgumentError("RBH iDMRG requires unit filling: N must equal L"))
    unitcell_size >= 1 || throw(ArgumentError("unitcell_size must be positive"))
    max_occupancy >= 1 || throw(ArgumentError("max_occupancy must be positive"))
    nsweeps >= 1 || throw(ArgumentError("nsweeps must be positive"))
    !isempty(maxdims) && all(>(0), maxdims) ||
        throw(ArgumentError("maxdims must contain positive bond dimensions"))
    isfinite(cutoff) && cutoff >= 0 ||
        throw(ArgumentError("cutoff must be finite and nonnegative"))
    nsteps_per_stage >= 3 && isodd(nsteps_per_stage) ||
        throw(ArgumentError("nsteps_per_stage must be an odd integer >= 3"))
    eigsolve_krylovdim >= 1 ||
        throw(ArgumentError("eigsolve_krylovdim must be positive"))

    finite_hamiltonian, bulk_hamiltonian, sites =
        _rabi_idmrg_hamiltonians(params, unitcell_size, max_occupancy)

    if init_state === nothing
        init_state = fill("0", 2 * unitcell_size)
        for i in 1:unitcell_size
            init_state[site_a(i)] = "1"
        end
    end
    length(init_state) == 2 * unitcell_size ||
        throw(ArgumentError("init_state must have one state per species mode in the unit cell"))
    occupancies = tryparse.(Int, init_state)
    all(n -> n !== nothing, occupancies) ||
        throw(ArgumentError("init_state entries must be nonnegative integer occupations"))
    occupancies_int = Int[something(n) for n in occupancies]
    all(n -> 0 <= n <= max_occupancy, occupancies_int) ||
        throw(ArgumentError("init_state occupations must be between 0 and max_occupancy"))
    sum(occupancies_int) == unitcell_size ||
        throw(ArgumentError("init_state must contain exactly unitcell_size particles"))
    psi_init = MPS(sites, init_state)
    infinite_hamiltonian = iDMRG.iMPO(
        finite_hamiltonian, bulk_hamiltonian, length(sites); site=2,
    )
    energy_observer = RabiIDMRGEnergyObserver(nothing)
    maxdim_schedule = [fill(maxdim, nsteps_per_stage) for maxdim in maxdims]

    psi = iDMRG.idmrg(
        iDMRG.iMPS(psi_init),
        infinite_hamiltonian;
        nstep_max=length(maxdims),
        nsteps=fill(nsteps_per_stage, length(maxdims)),
        nsweeps,
        maxdims=maxdim_schedule,
        cutoff,
        eigsolve_krylovdim,
        observer=energy_observer,
    )
    energy_observer.energy_density === nothing &&
        throw(ErrorException("iDMRG returned without reporting an energy density"))

    energy_per_site = 2 * energy_observer.energy_density
    if dmc_shift
        e_bot, e_top = mathieu_band_edges(params.V0, params.kL)
        energy_per_site += (e_bot + e_top) / 2
    end
    return energy_per_site, psi, sites, params
end

function two_component_idmrg_ground_state(
    N::Int, L::Int, J::Real, U_a::Real, U_b::Real, U_ab::Real,
    t::Real, h::Real, V0::Real, kL::Real;
    kwargs...,
)
    params = RabiParams(; N, L, J, U_a, U_b, U_ab, t, h, V0, kL)
    return two_component_idmrg_ground_state(params; kwargs...)
end
