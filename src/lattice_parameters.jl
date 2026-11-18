include(joinpath(@__DIR__, "wannier.jl"))
import .MathieuWannier
using LinearAlgebra

function compute_lattice_parameters(;
    L::Real,
    lattice_sites::Integer,
    V0::Real,
    g_up::Real,
    g_down::Real,
    g_updown::Real,
    hbar::Real=1.0,
    mass::Real=1.0,
    lattice_phase::Real=0.0,
    cutoff::Integer=30,
    points_per_site::Integer=max(128, 2 * cutoff + 1),
)
    isfinite(g_up) || throw(ArgumentError("g_up must be finite"))
    isfinite(g_down) || throw(ArgumentError("g_down must be finite"))
    isfinite(g_updown) || throw(ArgumentError("g_updown must be finite"))

    kL = π * lattice_sites / L
    E_r = hbar^2 * kL^2 / (2.0 * mass)
    V0_energy = V0 * E_r

    wannier = MathieuWannier.compute_wannier(
        V0_energy,
        lattice_sites,
        L;
        cutoff,
        points_per_site,
        lattice_phase,
        hbar,
        mass,
    )

    dx = wannier.dx
    points_per_site_int = Int(points_per_site)
    w_left = wannier.psi
    w_right = circshift(w_left, points_per_site_int)
    second_derivative = (
        .-circshift(w_right, -2) .+
        16.0 .* circshift(w_right, -1) .-
        30.0 .* w_right .+
        16.0 .* circshift(w_right, 1) .-
        circshift(w_right, 2)
    ) ./ (12.0 * dx^2)

    potential = V0_energy .* sin.(kL .* (wannier.x .- lattice_phase)) .^ 2

    h_w_right = (
        -(Float64(hbar)^2 / (2.0 * Float64(mass))) .* second_derivative .+
        potential .* w_right
    )
    hopping = -real(sum(conj.(w_left) .* h_w_right) * dx)

    onsite_overlap = sum(abs2.(w_left) .^ 2) * dx
    return (
        J=hopping,
        U_up=Float64(g_up) * onsite_overlap,
        U_down=Float64(g_down) * onsite_overlap,
        U_updown=Float64(g_updown) * onsite_overlap,
        onsite_overlap=onsite_overlap,
        wannier=wannier,
    )
end
