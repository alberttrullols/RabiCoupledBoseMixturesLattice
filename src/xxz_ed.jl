using Printf
using LinearAlgebra
using SparseArrays
using KrylovKit

isdefined(@__MODULE__, :RabiModel) || include(joinpath(@__DIR__, "models", "rabi_model.jl"))
using .RabiModel
isdefined(@__MODULE__, :XXZModel) || include(joinpath(@__DIR__, "models", "xxz_model.jl"))
using .XXZModel
include(joinpath(@__DIR__, "observables.jl"))


function spin_ground_state(params::SpinParams; include_xy::Bool=true, krylov::Bool=true)
    H = XXZModel.build_ed_hamiltonian(params; include_xy)
    dim = 2^params.L

    if !krylov && dim <= 500
        decomposition = eigen(Symmetric(Matrix(H)))
        E0 = decomposition.values[1]
        psi0 = decomposition.vectors[:, 1]
    else
        values, vectors, _ = eigsolve(Symmetric(H), dim, 1, :SR; ishermitian=true)
        E0 = real(values[1])
        psi0 = vectors[1]
    end
    psi0 = psi0 / norm(psi0)
    return E0, psi0
end


function xxz_ground_state(
    L::Int; Jxy::Real=0.0, Jz::Real, hx::Real=0.0, hz::Real=0.0,
    include_xy::Bool=true, krylov::Bool=true,
)
    params = SpinParams(; L, Jxy, Jz, hx, hz)
    E0, psi0 = spin_ground_state(params; include_xy, krylov)
    return E0, psi0, params
end


function xxz_ground_state_from_rbh(
    N::Int, L::Int, J::Real, U::Real, U_ab::Real, t::Real, h::Real=0.0;
    include_xy::Bool=true, krylov::Bool=true,
)
    params, C = rbh_to_xxz_params(N, L, J, U, U_ab, t, h)
    E_spin, psi0 = spin_ground_state(params; include_xy, krylov)
    E_total = E_spin + L * C
    return E_total, psi0, params, C
end
