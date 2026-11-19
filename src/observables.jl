using LinearAlgebra
using ITensors
using ITensorMPS

square_polarization_from_correlations(D::AbstractMatrix, N::Real) = sum(D) / N^2

spin_spin_correlations(D::AbstractMatrix) = D ./ 4

function structure_factor(D::AbstractMatrix, k::Real)
    L = size(D, 1)
    size(D) == (L, L) || throw(ArgumentError("D must be a square L x L matrix"))
    total = sum(D[i, j] * cis(k * (i - j)) for i in 1:L, j in 1:L)
    return real(total) / L
end

function structure_factor_spectrum(D::AbstractMatrix)
    L = size(D, 1)
    ks = [2π * m / L for m in 0:(L - 1)]
    Sk = [structure_factor(D, k) for k in ks]
    return ks, Sk
end

energy_per_particle(E::Real, N::Integer) = E / N

function polarization_moment(psi::AbstractVector{<:Number}, basis, N::Int, L::Int, order::Int)
    return sum(
        abs2(psi[i]) * ((2 * sum(basis[i][1:L]) - N) / N)^order
        for i in eachindex(basis)
    )
end

polarization(psi::AbstractVector{<:Number}, basis, N::Int, L::Int) =
    polarization_moment(psi, basis, N, L, 1)
square_polarization(psi::AbstractVector{<:Number}, basis, N::Int, L::Int) =
    polarization_moment(psi, basis, N, L, 2)
quartic_polarization(psi::AbstractVector{<:Number}, basis, N::Int, L::Int) =
    polarization_moment(psi, basis, N, L, 4)


function polarization_correlation_matrix(psi::AbstractVector{<:Number}, basis, L::Int)
    D = zeros(Float64, L, L)
    @inbounds for alpha in eachindex(basis)
        p = abs2(psi[alpha])
        p == 0 && continue
        state = basis[alpha]
        for i in 1:L
            di = state[i] - state[L + i]
            di == 0 && continue
            for j in 1:L
                dj = state[j] - state[L + j]
                D[i, j] += p * di * dj
            end
        end
    end
    return D
end


function transverse_polarization(psi::AbstractVector{<:Number}, basis, N::Int, L::Int)
    index = Dict(s => i for (i, s) in enumerate(basis))
    total = zero(eltype(psi))
    @inbounds for (alpha, state) in enumerate(basis)
        ca = psi[alpha]
        ca == 0 && continue
        for i in 1:L, (src, dst) in ((i, L + i), (L + i, i))
            result = hop_matrix_element(state, src, dst)
            result === nothing && continue
            new_state, amplitude = result
            beta = get(index, new_state, nothing)
            beta === nothing && continue
            total += conj(psi[beta]) * amplitude * ca
        end
    end
    return real(total) / N
end


energy_per_particle(H::MPO, psi::MPS, N::Integer) = real(inner(psi', H, psi)) / N

function polarization(psi::MPS, p::RabiParams)
    n = expect(psi, "N")
    return sum(n[site_a(i)] - n[site_b(i)] for i in 1:p.L) / p.N
end


function polarization_correlation_matrix(psi::MPS, p::RabiParams)
    C = correlation_matrix(psi, "N", "N")
    L = p.L
    D = zeros(Float64, L, L)
    for i in 1:L, j in 1:L
        D[i, j] = real(
            C[site_a(i), site_a(j)] - C[site_a(i), site_b(j)] -
            C[site_b(i), site_a(j)] + C[site_b(i), site_b(j)]
        )
    end
    return D
end

square_polarization(psi::MPS, p::RabiParams) =
    square_polarization_from_correlations(polarization_correlation_matrix(psi, p), p.N)

function polarization_mpo(p::RabiParams, sites)
    os = OpSum()
    for i in 1:p.L
        os += 1 / p.N, "N", site_a(i)
        os += -1 / p.N, "N", site_b(i)
    end
    return MPO(os, sites)
end

function quartic_polarization(
    psi::MPS, p::RabiParams, sites; cutoff::Real=1e-12, maxdim::Int=typemax(Int),
)
    P = polarization_mpo(p, sites)
    chi = apply(P, psi; cutoff=cutoff, maxdim=maxdim)
    chi = apply(P, chi; cutoff=cutoff, maxdim=maxdim)
    return real(inner(chi, chi))
end


function transverse_polarization(psi::MPS, p::RabiParams)
    C = correlation_matrix(psi, "Adag", "A")
    total = sum(real(C[site_a(i), site_b(i)] + C[site_b(i), site_a(i)]) for i in 1:p.L)
    return total / p.N
end


function polarization_moment(psi::AbstractVector{<:Number}, L::Int, order::Int)
    dim = 2^L
    length(psi) == dim || throw(ArgumentError("psi length must be 2^L"))
    return sum(
        abs2(psi[state + 1]) * ((2 * count_ones(state) - L) / L)^order
        for state in 0:(dim - 1)
    )
end

polarization(psi::AbstractVector{<:Number}, L::Int) = polarization_moment(psi, L, 1)
square_polarization(psi::AbstractVector{<:Number}, L::Int) = polarization_moment(psi, L, 2)
quartic_polarization(psi::AbstractVector{<:Number}, L::Int) = polarization_moment(psi, L, 4)

function polarization_correlation_matrix(psi::AbstractVector{<:Number}, L::Int)
    dim = 2^L
    length(psi) == dim || throw(ArgumentError("psi length must be 2^L"))
    D = zeros(Float64, L, L)
    @inbounds for state in 0:(dim - 1)
        p = abs2(psi[state + 1])
        p == 0 && continue
        for i in 1:L
            si = ((state >> (i - 1)) & 1) == 1 ? 1.0 : -1.0
            for j in 1:L
                sj = ((state >> (j - 1)) & 1) == 1 ? 1.0 : -1.0
                D[i, j] += p * si * sj
            end
        end
    end
    return D
end

function transverse_polarization(psi::AbstractVector{<:Number}, L::Int)
    dim = 2^L
    length(psi) == dim || throw(ArgumentError("psi length must be 2^L"))
    total = zero(eltype(psi))
    @inbounds for state in 0:(dim - 1)
        ca = psi[state + 1]
        ca == 0 && continue
        for i in 1:L
            flipped = xor(state, 1 << (i - 1))
            total += conj(psi[flipped + 1]) * ca
        end
    end
    return real(total) / L
end

function polarization(psi::MPS, L::Int)
    n = expect(psi, "Sz")
    return 2 * sum(n) / L
end

function polarization_correlation_matrix(psi::MPS, L::Int)
    C = correlation_matrix(psi, "Sz", "Sz")
    return 4 .* real.(C)
end

square_polarization(psi::MPS, L::Int) =
    square_polarization_from_correlations(polarization_correlation_matrix(psi, L), L)

function polarization_mpo(L::Int, sites)
    os = OpSum()
    for i in 1:L
        os += 2 / L, "Sz", i
    end
    return MPO(os, sites)
end

function quartic_polarization(
    psi::MPS, L::Int, sites; cutoff::Real=1e-12, maxdim::Int=typemax(Int),
)
    P = polarization_mpo(L, sites)
    chi = apply(P, psi; cutoff=cutoff, maxdim=maxdim)
    chi = apply(P, chi; cutoff=cutoff, maxdim=maxdim)
    return real(inner(chi, chi))
end

function transverse_polarization(psi::MPS, L::Int)
    n = expect(psi, "Sx")
    return 2 * sum(n) / L
end
