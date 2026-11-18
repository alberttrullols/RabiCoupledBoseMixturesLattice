module MathieuWannier

using LinearAlgebra

export WannierResult, compute_wannier

struct WannierResult
    x::Vector{Float64}
    psi::Vector{ComplexF64}
    q::Vector{Float64}
    band_energies::Vector{Float64}
    dx::Float64
    lattice_spacing::Float64
end

function compute_wannier(
    V0::Real,
    lattice_sites::Integer,
    L::Real;
    cutoff::Integer=30,
    points_per_site::Integer=128,
    site::Integer=0,
    lattice_phase::Real=0.0,
    hbar::Real=1.0,
    mass::Real=1.0,
)
    isfinite(V0) && V0 > 0 ||
        throw(ArgumentError("V0 must be finite and positive"))
    lattice_sites >= 2 ||
        throw(ArgumentError("lattice_sites must be at least 2"))
    isfinite(L) && L > 0 ||
        throw(ArgumentError("L must be finite and positive"))
    isfinite(lattice_phase) ||
        throw(ArgumentError("lattice_phase must be finite"))
    isfinite(hbar) && hbar > 0 ||
        throw(ArgumentError("hbar must be finite and positive"))
    isfinite(mass) && mass > 0 ||
        throw(ArgumentError("mass must be finite and positive"))
    cutoff >= 1 || throw(ArgumentError("cutoff must be at least 1"))
    points_per_site >= 2 * cutoff + 1 ||
        throw(ArgumentError("points_per_site must be at least 2*cutoff + 1"))

    V = Float64(V0)
    box_length = Float64(L)
    phase = Float64(lattice_phase)
    kinetic_coefficient = Float64(hbar)^2 / (2.0 * Float64(mass))
    nsites = Int(lattice_sites)
    M = Int(cutoff)
    spacing = box_length / nsites
    kL = π / spacing
    reciprocal_lattice_vector = 2.0 * kL
    plane_wave_indices = collect(-M:M)
    nq = nsites

    # Allowed periodic-box quasimomenta, sorted across the first Brillouin zone.
    q_indices = collect(-fld(nq, 2):(cld(nq, 2) - 1))
    q_values = (2.0 * π / box_length) .* q_indices
    coefficients = Matrix{ComplexF64}(undef, length(plane_wave_indices), nq)
    band_energies = Vector{Float64}(undef, nq)
    off_diagonal = fill(-V / 4.0, 2M)

    for iq in eachindex(q_values)
        q = q_values[iq]
        diagonal = [
            kinetic_coefficient * (q + reciprocal_lattice_vector * n)^2 + V / 2.0
            for n in plane_wave_indices
        ]
        decomposition = eigen(SymTridiagonal(diagonal, off_diagonal))
        band_energies[iq] = decomposition.values[1]
        coefficients[:, iq] = decomposition.vectors[:, 1]
    end

    # Parallel-transport gauge: choose adjacent Bloch states to have positive
    # overlap, then spread the remaining Berry phase uniformly around the zone.
    for iq in 1:(nq - 1)
        overlap = dot(view(coefficients, :, iq), view(coefficients, :, iq + 1))
        abs(overlap) > eps(Float64) ||
            throw(ErrorException("Bloch-band overlap vanished; increase lattice_sites or cutoff"))
        coefficients[:, iq + 1] .*= cis(-angle(overlap))
    end

    # q = q_min + reciprocal_lattice_vector is equivalent to q = q_min with
    # its plane-wave indices shifted by one.
    shifted_first = zeros(ComplexF64, length(plane_wave_indices))
    shifted_first[1:(end - 1)] = coefficients[2:end, 1]
    zone_overlap = dot(view(coefficients, :, nq), shifted_first)
    abs(zone_overlap) > eps(Float64) ||
        throw(ErrorException("Brillouin-zone boundary overlap vanished; increase cutoff"))
    berry_phase = angle(zone_overlap)
    for iq in 1:nq
        coefficients[:, iq] .*= cis((iq - 1) * berry_phase / nq)
    end

    npoints = nsites * Int(points_per_site)
    dx = box_length / npoints
    x = [-box_length / 2.0 + (ix - 1) * dx for ix in 1:npoints]
    center = phase + Float64(site) * spacing
    psi = zeros(ComplexF64, npoints)
    normalization = sqrt(nq * box_length)

    @inbounds for ix in eachindex(x)
        position = x[ix]
        value = 0.0 + 0.0im
        for iq in eachindex(q_values)
            periodic_part = 0.0 + 0.0im
            for (icoeff, n) in enumerate(plane_wave_indices)
                periodic_part += coefficients[icoeff, iq] *
                    cis(reciprocal_lattice_vector * n * (position - phase))
            end
            value += cis(q_values[iq] * (position - center)) * periodic_part
        end
        psi[ix] = value / normalization
    end

    center_in_box = mod(center + box_length / 2.0, box_length) - box_length / 2.0
    center_index = argmin(abs.(x .- center_in_box))
    psi .*= cis(-angle(psi[center_index]))
    sampled_norm = sqrt(sum(abs2, psi) * dx)
    isfinite(sampled_norm) && sampled_norm > 0 ||
        throw(ErrorException("computed Wannier function has invalid norm"))
    psi ./= sampled_norm

    return WannierResult(x, psi, q_values, band_energies, dx, spacing)
end

end # module
