module RabiModel

using LinearAlgebra
using SparseArrays
using ITensors
using ITensorMPS

export RabiParams, mathieu_band_edges, build_two_species_basis,
       site_a, site_b, hop_matrix_element, hamiltonian_terms

struct RabiParams
    N::Int
    L::Int
    J::Float64
    U_a::Float64
    U_b::Float64
    U_ab::Float64
    t::Float64
    h::Float64
    V0::Float64
    kL::Float64
end

function RabiParams(;
    N::Integer, L::Integer, J::Real, U_a::Real, U_b::Real, U_ab::Real,
    t::Real=0.0, h::Real=0.0, V0::Real=0.0, kL::Real=π,
)
    validate_params(N, L, J, U_a, U_b, U_ab)
    return RabiParams(
        Int(N), Int(L), Float64(J), Float64(U_a), Float64(U_b),
        Float64(U_ab), Float64(t), Float64(h), Float64(V0), Float64(kL),
    )
end

function validate_params(N::Integer, L::Integer, J::Real, U_a::Real, U_b::Real, U_ab::Real)
    L >= 2 || throw(ArgumentError("Need at least L=2 sites."))
    N >= 1 || throw(ArgumentError("Need at least one boson."))
    J > 0 || throw(ArgumentError("J must be positive."))
    (U_a >= 0 && U_b >= 0 && U_ab >= 0) || throw(ArgumentError("Repulsive U >= 0 assumed."))
    return nothing
end



function ring_bonds(L::Integer)
    L >= 2 || throw(ArgumentError("Need at least L=2 sites."))
    return L == 2 ? [(1, 2)] : [(j, mod1(j + 1, L)) for j in 1:L]
end



function mathieu_band_edges(V0::Real, kL::Real; M::Int=30)
    n = -M:M
    edges = map((0.0, Float64(kL))) do q
        diagonal = [(2n_i * kL + q)^2 / 2 + V0 / 2 for n_i in n]
        eigmin(SymTridiagonal(diagonal, fill(-V0 / 4, 2M)))
    end
    return edges
end



function build_two_species_basis(N::Int, L::Int; N1::Union{Nothing,Int}=nothing)
    basis = Vector{NTuple{2L,Int}}()
    state = zeros(Int, 2L)
    function rec(site, remaining)
        if site == 2L
            state[2L] = remaining
            push!(basis, ntuple(i -> state[i], 2L))
            return
        end
        for n in remaining:-1:0
            state[site] = n
            rec(site + 1, remaining - n)
        end
    end
    rec(1, N)
    N1 === nothing || filter!(s -> sum(s[1:L]) == N1, basis)
    return basis
end

# Interleaved DMRG site convention: site 2i-1 = species a at physical site
# i, site 2i = species b at physical site i, so the Rabi term is a
# nearest-neighbor "rung" bond.
site_a(i::Integer) = 2i - 1
site_b(i::Integer) = 2i



function hop_matrix_element(state::NTuple{M,Int}, src::Int, dst::Int) where {M}
    state[src] > 0 || return nothing
    new = collect(state)
    new[src] -= 1
    new[dst] += 1
    amplitude = sqrt(state[src] * (state[dst] + 1))
    return Tuple(new), amplitude
end

# One term of the Rabi-Bose-Hubbard Hamiltonian, representation-agnostic:
# both build_ed_hamiltonian and build_mpo_hamiltonian below consume the
# same term list.
struct HTerm
    kind::Symbol               # :onsite_a, :onsite_b, :onsite_ab, :bias, :hop_a, :hop_b, :rabi
    idx::Tuple{Vararg{Int}}    # site index/indices (1-based, on the L-site ring)
    coeff::Float64
end

function hamiltonian_terms(p::RabiParams; boundary::Symbol=:periodic)
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    terms = HTerm[]
    for i in 1:p.L
        p.U_a  != 0 && push!(terms, HTerm(:onsite_a,  (i,), p.U_a))
        p.U_b  != 0 && push!(terms, HTerm(:onsite_b,  (i,), p.U_b))
        p.U_ab != 0 && push!(terms, HTerm(:onsite_ab, (i,), p.U_ab))
        p.h    != 0 && push!(terms, HTerm(:bias,      (i,), p.h))
        p.t    != 0 && push!(terms, HTerm(:rabi,      (i,), p.t))
    end
    bonds = boundary == :periodic ? ring_bonds(p.L) : [(i, i + 1) for i in 1:(p.L - 1)]
    for (i, j) in bonds
        push!(terms, HTerm(:hop_a, (i, j), p.J))
        push!(terms, HTerm(:hop_b, (i, j), p.J))
    end
    return terms
end

# Exact-diagonalization Hamiltonian (sparse matrix on the Fock basis)

function build_ed_hamiltonian(p::RabiParams; N1::Union{Nothing,Int}=nothing)
    if N1 !== nothing
        p.t == 0 || throw(ArgumentError("N1 can only be fixed when t = 0"))
        0 <= N1 <= p.N || throw(ArgumentError("N1 must be between 0 and N"))
    end

    basis = build_two_species_basis(p.N, p.L; N1=N1)
    dim = length(basis)
    index = Dict(s => i for (i, s) in enumerate(basis))
    terms = hamiltonian_terms(p)

    rows = Int[]; cols = Int[]; vals = Float64[]
    function add!(r, c, v)
        push!(rows, r); push!(cols, c); push!(vals, v)
    end

    L = p.L
    for (alpha, state) in enumerate(basis)
        for term in terms
            _apply_ed_term!(add!, term, state, alpha, index, L)
        end
    end
    H = sparse(rows, cols, vals, dim, dim)
    return H, basis
end

function _apply_ed_term!(add!, term::HTerm, state, alpha::Int, index, L::Int)
    kind, idx, coeff = term.kind, term.idx, term.coeff
    if kind == :onsite_a
        i, = idx
        n = state[i]
        add!(alpha, alpha, 0.5 * coeff * n * (n - 1))
    elseif kind == :onsite_b
        i, = idx
        n = state[L + i]
        add!(alpha, alpha, 0.5 * coeff * n * (n - 1))
    elseif kind == :onsite_ab
        i, = idx
        add!(alpha, alpha, coeff * state[i] * state[L + i])
    elseif kind == :bias
        i, = idx
        add!(alpha, alpha, -coeff * (state[i] - state[L + i]))
    elseif kind == :hop_a
        i, j = idx
        _apply_hop!(add!, state, alpha, index, i, j, coeff)
        _apply_hop!(add!, state, alpha, index, j, i, coeff)
    elseif kind == :hop_b
        i, j = idx
        _apply_hop!(add!, state, alpha, index, L + i, L + j, coeff)
        _apply_hop!(add!, state, alpha, index, L + j, L + i, coeff)
    elseif kind == :rabi
        i, = idx
        _apply_hop!(add!, state, alpha, index, i, L + i, coeff)
        _apply_hop!(add!, state, alpha, index, L + i, i, coeff)
    else
        throw(ArgumentError("Unknown Hamiltonian term kind $(kind)"))
    end
    return nothing
end

function _apply_hop!(add!, state, alpha::Int, index, src::Int, dst::Int, coeff::Real)
    result = hop_matrix_element(state, src, dst)
    result === nothing && return nothing
    new_state, amplitude = result
    add!(index[new_state], alpha, -coeff * amplitude)
    return nothing
end

# DMRG Hamiltonian (MPO on an interleaved 2L-site chain)

function build_mpo_hamiltonian(p::RabiParams, sites; boundary::Symbol=:periodic)
    length(sites) == 2p.L ||
        throw(ArgumentError("Expected 2L = $(2p.L) sites, got $(length(sites))"))
    os = OpSum()
    for term in hamiltonian_terms(p; boundary)
        os = _add_mpo_term!(os, term)
    end
    return MPO(os, sites)
end

function _add_mpo_term!(os::OpSum, term::HTerm)
    kind, idx, coeff = term.kind, term.idx, term.coeff
    if kind == :onsite_a
        i, = idx
        a = site_a(i)
        os += coeff / 2, "N", a, "N", a
        os += -coeff / 2, "N", a
    elseif kind == :onsite_b
        i, = idx
        b = site_b(i)
        os += coeff / 2, "N", b, "N", b
        os += -coeff / 2, "N", b
    elseif kind == :onsite_ab
        i, = idx
        os += coeff, "N", site_a(i), "N", site_b(i)
    elseif kind == :bias
        i, = idx
        os += -coeff, "N", site_a(i)
        os += coeff, "N", site_b(i)
    elseif kind == :hop_a
        i, j = idx
        os += -coeff, "Adag", site_a(i), "A", site_a(j)
        os += -coeff, "Adag", site_a(j), "A", site_a(i)
    elseif kind == :hop_b
        i, j = idx
        os += -coeff, "Adag", site_b(i), "A", site_b(j)
        os += -coeff, "Adag", site_b(j), "A", site_b(i)
    elseif kind == :rabi
        i, = idx
        os += -coeff, "Adag", site_a(i), "A", site_b(i)
        os += -coeff, "Adag", site_b(i), "A", site_a(i)
    else
        throw(ArgumentError("Unknown Hamiltonian term kind $(kind)"))
    end
    return os
end

end # module
