using Printf
using ITensors
using ITensorMPS

isdefined(@__MODULE__, :RabiModel) || include(joinpath(@__DIR__, "models", "rabi_model.jl"))
using .RabiModel
isdefined(@__MODULE__, :XXZModel) || include(joinpath(@__DIR__, "models", "xxz_model.jl"))
using .XXZModel
include(joinpath(@__DIR__, "observables.jl"))

function spin_dmrg_ground_state(
    params::SpinParams; include_xy::Bool=true,
    nsweeps::Int=10, maxdims=[10, 20, 50, 100, 200], cutoff::Real=1e-10,
    init_state::Union{Nothing,Vector{String}}=nothing,
)
    sites = siteinds("S=1/2", params.L; conserve_qns=false)
    H = XXZModel.build_mpo_hamiltonian(params, sites; include_xy)

    if init_state === nothing
        # Neel-like default guess; DMRG relaxes it from there.
        init_state = [isodd(i) ? "Up" : "Dn" for i in 1:params.L]
    end
    psi_init = random_mps(sites, init_state; linkdims=4)

    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdims...)
    setcutoff!(sweeps, cutoff)

    E0, psi = dmrg(H, psi_init, sweeps; outputlevel=0)
    return E0, psi, sites
end

function xxz_dmrg_ground_state(
    L::Int; Jxy::Real=0.0, Jz::Real, hx::Real=0.0, hz::Real=0.0,
    include_xy::Bool=true, kwargs...,
)
    params = SpinParams(; L, Jxy, Jz, hx, hz)
    E0, psi, sites = spin_dmrg_ground_state(params; include_xy, kwargs...)
    return E0, psi, sites, params
end

function xxz_dmrg_ground_state_from_rbh(
    N::Int, L::Int, J::Real, U::Real, U_ab::Real, t::Real, h::Real=0.0;
    include_xy::Bool=true, kwargs...,
)
    params, C = rbh_to_xxz_params(N, L, J, U, U_ab, t, h)
    E_spin, psi, sites = spin_dmrg_ground_state(params; include_xy, kwargs...)
    E_total = E_spin + L * C
    return E_total, psi, sites, params, C
end
