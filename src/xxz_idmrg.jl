using ITensors
using ITensorMPS
import iDMRG

isdefined(@__MODULE__, :XXZModel) || include(joinpath(@__DIR__, "models", "xxz_model.jl"))
using .XXZModel

mutable struct SpinIDMRGEnergyObserver
    energy_density::Union{Nothing,Float64}
end

function iDMRG.checkdone!(observer::SpinIDMRGEnergyObserver; eng_density, kwargs...)
    observer.energy_density = Float64(eng_density)
    return false
end

function _idmrg_hamiltonians(
    params::SpinParams, include_xy::Bool, unitcell_size::Int,
)
    nsites = 6 * unitcell_size
    finite_sites = siteinds("S=1/2", nsites; conserve_qns=false)
    os = OpSum()

    for i in 1:(nsites - 1)
        if params.Jz != 0
            os += params.Jz, "Sz", i, "Sz", i + 1
        end
        if include_xy && params.Jxy != 0
            os += params.Jxy / 2, "S+", i, "S-", i + 1
            os += params.Jxy / 2, "S-", i, "S+", i + 1
        end
    end
    for i in 1:nsites
        if params.hx != 0
            os += -params.hx, "Sx", i
        end
        if params.hz != 0
            os += -params.hz, "Sz", i
        end
    end

    finite_hamiltonian = MPO(os, finite_sites)
    cell_sites = siteinds("S=1/2", unitcell_size; conserve_qns=false)
    bulk_hamiltonian = iDMRG.initializeHam(
        finite_hamiltonian, unitcell_size, cell_sites;
        site_start=3 * unitcell_size, ham_uc=1,
    )
    return finite_hamiltonian, bulk_hamiltonian, cell_sites
end


function spin_idmrg_ground_state(
    params::SpinParams;
    include_xy::Bool=true,
    nsweeps::Int=10,
    maxdims::Vector{Int}=[10, 20, 50, 100, 200],
    cutoff::Real=1e-10,
    unitcell_size::Int=params.L,
    nsteps_per_stage::Int=3,
    eigsolve_krylovdim::Int=3,
    init_state::Union{Nothing,Vector{String}}=nothing,
)
    unitcell_size >= 2 && iseven(unitcell_size) ||
        throw(ArgumentError("unitcell_size must be a positive even integer >= 2"))
    nsweeps >= 1 || throw(ArgumentError("nsweeps must be positive"))
    !isempty(maxdims) && all(>(0), maxdims) ||
        throw(ArgumentError("maxdims must contain positive bond dimensions"))
    isfinite(cutoff) && cutoff >= 0 ||
        throw(ArgumentError("cutoff must be finite and nonnegative"))
    nsteps_per_stage >= 3 && isodd(nsteps_per_stage) ||
        throw(ArgumentError("nsteps_per_stage must be an odd integer >= 3"))
    eigsolve_krylovdim >= 1 ||
        throw(ArgumentError("eigsolve_krylovdim must be positive"))

    cell_params = SpinParams(;
        L=unitcell_size, Jxy=params.Jxy, Jz=params.Jz, hx=params.hx, hz=params.hz,
    )
    finite_hamiltonian, bulk_hamiltonian, sites =
        _idmrg_hamiltonians(cell_params, include_xy, unitcell_size)

    if init_state === nothing
        init_state = [isodd(i) ? "Up" : "Dn" for i in 1:unitcell_size]
    end
    length(init_state) == unitcell_size ||
        throw(ArgumentError("init_state must have one state per unit-cell site"))
    psi_init = MPS(sites, init_state)
    infinite_hamiltonian = iDMRG.iMPO(
        finite_hamiltonian, bulk_hamiltonian, unitcell_size; site=2,
    )
    energy_observer = SpinIDMRGEnergyObserver(nothing)
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
    return energy_observer.energy_density, psi, sites, cell_params
end

function xxz_idmrg_ground_state(;
    unitcell_size::Int=2,
    Jxy::Real=0.0,
    Jz::Real,
    hx::Real=0.0,
    hz::Real=0.0,
    include_xy::Bool=true,
    kwargs...,
)
    params = SpinParams(; L=unitcell_size, Jxy, Jz, hx, hz)
    energy, psi, sites, params =
        spin_idmrg_ground_state(params; include_xy, unitcell_size, kwargs...)
    return energy, psi, sites, params
end

function tfim_idmrg_ground_state(; Jz::Real, kwargs...)
    return xxz_idmrg_ground_state(; Jxy=0.0, Jz, include_xy=false, kwargs...)
end

function xxz_idmrg_ground_state_from_rbh(
    N::Int, L::Int, J::Real, U::Real, U_ab::Real, t::Real, h::Real=0.0;
    include_xy::Bool=true, unitcell_size::Int=2, kwargs...,
)
    mapped_params, C = rbh_to_xxz_params(N, L, J, U, U_ab, t, h)
    params = SpinParams(;
        L=unitcell_size, Jxy=mapped_params.Jxy, Jz=mapped_params.Jz,
        hx=mapped_params.hx, hz=mapped_params.hz,
    )
    energy, psi, sites, params =
        spin_idmrg_ground_state(params; include_xy, unitcell_size, kwargs...)
    return energy + C, psi, sites, params, C
end

function tfim_idmrg_ground_state_from_rbh(args...; kwargs...)
    return xxz_idmrg_ground_state_from_rbh(args...; include_xy=false, kwargs...)
end
