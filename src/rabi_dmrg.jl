using Printf
using ITensors
using ITensorMPS

isdefined(@__MODULE__, :RabiModel) || include(joinpath(@__DIR__, "models", "rabi_model.jl"))
using .RabiModel
include(joinpath(@__DIR__, "observables.jl"))

function two_component_dmrg_ground_state(
    N::Int, L::Int, J::Real, U_a::Real, U_b::Real, U_ab::Real,
    t::Real, h::Real, V0::Real, kL::Real;
    dmc_shift::Bool=false,
    nsweeps::Int=10,
    maxdims=[10, 20, 50, 100, 200],
    cutoff::Real=1e-10,
    init_state::Union{Nothing,Vector{String}}=nothing,
)
    params = RabiParams(; N, L, J, U_a, U_b, U_ab, t, h, V0, kL)

    nsites = 2L
    sites = siteinds("Boson", nsites; dim=N + 1, conserve_qns=true)
    H = RabiModel.build_mpo_hamiltonian(params, sites)

    if init_state === nothing
        # Spread the N bosons across species-a sites of distinct physical
        # sites for the initial product state (DMRG relaxes it from
        # there); piles onto site 1 once N > L.
        state = fill("0", nsites)
        for k in 1:N
            idx = site_a(mod1(k, L))
            state[idx] = string(parse(Int, state[idx]) + 1)
        end
        init_state = state
    end
    psi_init = random_mps(sites, init_state; linkdims=4)

    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdims...)
    setcutoff!(sweeps, cutoff)

    E0, psi = dmrg(H, psi_init, sweeps; outputlevel=0)

    e_bot, e_top = mathieu_band_edges(V0, kL)
    E_total = dmc_shift ? E0 + N * (e_bot + e_top) / 2 : E0
    return E_total, psi, sites, params
end
