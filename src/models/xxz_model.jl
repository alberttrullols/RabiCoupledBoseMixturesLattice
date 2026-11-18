module XXZModel

using LinearAlgebra
using SparseArrays
using ITensors
using ITensorMPS

export SpinParams, rbh_to_xxz_params, spin_hamiltonian_terms

struct SpinParams
    L::Int
    Jxy::Float64
    Jz::Float64
    hx::Float64
    hz::Float64
end

function SpinParams(; L::Integer, Jxy::Real=0.0, Jz::Real, hx::Real=0.0, hz::Real=0.0)
    L >= 2 || throw(ArgumentError("Need at least L=2 sites."))
    return SpinParams(Int(L), Float64(Jxy), Float64(Jz), Float64(hx), Float64(hz))
end

function rbh_to_xxz_params(N::Integer, L::Integer, J::Real, U::Real, U_ab::Real, t::Real, h::Real=0.0)
    N == L || throw(ArgumentError("The spin-1/2 mapping assumes unit filling N == L."))
    J > 0 || throw(ArgumentError("J must be positive"))
    U > 0 || throw(ArgumentError("U must be positive"))
    U_ab > 0 || throw(ArgumentError("U_ab must be positive"))

    Jxy = -4 * J^2 / U_ab
    Jz = 4 * J^2 * (1 / U_ab - 2 / U)
    hx = 2 * t
    hz = 2 * h
    C = -2 * J^2 / U - J^2 / U_ab

    return SpinParams(; L, Jxy, Jz, hx, hz), C
end

function ring_bonds(L::Integer)
    L >= 2 || throw(ArgumentError("Need at least L=2 sites."))
    return L == 2 ? [(1, 2)] : [(j, mod1(j + 1, L)) for j in 1:L]
end

# One term of the spin Hamiltonian, representation-agnostic: both
# build_ed_hamiltonian and build_mpo_hamiltonian below consume the same
# term list.
struct SpinHTerm
    kind::Symbol               # :zz, :xy (bonds); :x, :z (sites)
    idx::Tuple{Vararg{Int}}
    coeff::Float64
end

function spin_hamiltonian_terms(p::SpinParams; include_xy::Bool=true)
    terms = SpinHTerm[]
    for (i, j) in ring_bonds(p.L)
        p.Jz != 0 && push!(terms, SpinHTerm(:zz, (i, j), p.Jz))
        include_xy && p.Jxy != 0 && push!(terms, SpinHTerm(:xy, (i, j), p.Jxy))
    end
    for i in 1:p.L
        p.hx != 0 && push!(terms, SpinHTerm(:x, (i,), p.hx))
        p.hz != 0 && push!(terms, SpinHTerm(:z, (i,), p.hz))
    end
    return terms
end


function build_ed_hamiltonian(p::SpinParams; include_xy::Bool=true)
    L = p.L
    dim = 2^L
    terms = spin_hamiltonian_terms(p; include_xy)

    rows = Int[]; cols = Int[]; vals = Float64[]
    function add!(r, c, v)
        push!(rows, r); push!(cols, c); push!(vals, v)
    end

    for state in 0:(dim - 1)
        diag = 0.0
        for term in terms
            if term.kind == :zz
                i, j = term.idx
                si = ((state >> (i - 1)) & 1) == 1 ? 0.5 : -0.5
                sj = ((state >> (j - 1)) & 1) == 1 ? 0.5 : -0.5
                diag += term.coeff * si * sj
            elseif term.kind == :z
                i, = term.idx
                si = ((state >> (i - 1)) & 1) == 1 ? 0.5 : -0.5
                diag -= term.coeff * si
            end
        end
        add!(state + 1, state + 1, diag)

        for term in terms
            if term.kind == :x
                i, = term.idx
                flipped = xor(state, 1 << (i - 1))
                add!(flipped + 1, state + 1, -term.coeff / 2)
            elseif term.kind == :xy
                i, j = term.idx
                si_up = ((state >> (i - 1)) & 1) == 1
                sj_up = ((state >> (j - 1)) & 1) == 1
                if si_up != sj_up
                    flipped = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                    add!(flipped + 1, state + 1, term.coeff / 2)
                end
            end
        end
    end
    H = sparse(rows, cols, vals, dim, dim)
    return H
end

# ---------------------------------------------------------------------
# DMRG Hamiltonian (MPO on an L-site "S=1/2" chain)
# ---------------------------------------------------------------------

"""
    build_mpo_hamiltonian(p::SpinParams, sites; include_xy=true)

Spin-1/2 XXZ(+field) Hamiltonian as an MPO on an `L`-site `"S=1/2"` chain.
"""
function build_mpo_hamiltonian(p::SpinParams, sites; include_xy::Bool=true)
    length(sites) == p.L ||
        throw(ArgumentError("Expected L = $(p.L) sites, got $(length(sites))"))
    os = OpSum()
    for term in spin_hamiltonian_terms(p; include_xy)
        kind, idx, coeff = term.kind, term.idx, term.coeff
        if kind == :zz
            i, j = idx
            os += coeff, "Sz", i, "Sz", j
        elseif kind == :xy
            i, j = idx
            os += coeff / 2, "S+", i, "S-", j
            os += coeff / 2, "S-", i, "S+", j
        elseif kind == :x
            i, = idx
            os += -coeff, "Sx", i
        elseif kind == :z
            i, = idx
            os += -coeff, "Sz", i
        end
    end
    return MPO(os, sites)
end

end # module
