using Printf
using LinearAlgebra
using SparseArrays
using KrylovKit

isdefined(@__MODULE__, :RabiModel) || include(joinpath(@__DIR__, "models", "rabi_model.jl"))
using .RabiModel
include(joinpath(@__DIR__, "observables.jl"))

function two_component_ground_state(
    N::Int, L::Int, J::Real, U_a::Real, U_b::Real, U_ab::Real,
    t::Real, h::Real, V0::Real, kL::Real;
    N1::Union{Nothing,Int}=nothing,
    dmc_shift::Bool=false,
)
    params = RabiParams(; N, L, J, U_a, U_b, U_ab, t, h, V0, kL)
    H, basis = RabiModel.build_ed_hamiltonian(params; N1=N1)
    dim = length(basis)

    if dim <= 500
        decomposition = eigen(Symmetric(Matrix(H)))
        E0 = decomposition.values[1]
        psi0 = decomposition.vectors[:, 1]
    else
        values, vectors, _ = eigsolve(Symmetric(H), dim, 1, :SR; ishermitian=true)
        E0 = values[1]
        psi0 = vectors[1]
    end
    psi0 = psi0 / norm(psi0)

    e_bot, e_top = mathieu_band_edges(V0, kL)
    E_total = dmc_shift ? E0 + N * (e_bot + e_top) / 2 : E0
    return E_total, psi0, basis
end
