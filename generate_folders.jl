using Printf

include(joinpath(@__DIR__, "constants.jl"))
include(joinpath(@__DIR__, "src", "models", "rabi_model.jl"))
using .RabiModel
include(joinpath(@__DIR__, "src", "models", "xxz_model.jl"))
using .XXZModel
include(joinpath(@__DIR__, "src", "lattice_parameters.jl"))

"""
    generate_folders(; model=MODEL_RABI, solver, parametrization=PARAM_LATTICE,
                      coupling_source=COUPLING_RBH,
                      size_pairs, L_values,                     # (N,L) pairs, or spin-native L sweep
                      t_values, h_values=[0.0], V0_values=[0.0], kL=π,
                      J_values, U_values, U_ab_values,           # parametrization = "lattice"
                      g_values, g12_values,                      # parametrization = "microscopic"
                      Jxy_values, Jz_values, hx_values, hz_values,   # coupling_source = "native"
                      wannier_cutoff=12, wannier_points_per_site=500,
                      N1_values=Int[], dmc_shift=false,
                      prefix="run", parent_folder="",
                      nsweeps=10, maxdims=[10,20,50,100,200], cutoff=1e-10)

Create one subfolder per combination of the swept parameters, each
containing a self-describing `params.dat` text file. `model` (one of
`MODELS`) and `solver` (one of `SOLVERS`) are single choices applied to
every folder in this call; call `generate_folders` again to also
generate folders for other models/solvers.

# Model selection (`model`)

  - `MODEL_RABI` (default): the full two-component Rabi-coupled
    Bose-Hubbard ring. Uses `size_pairs`, `parametrization`,
    `J_values`/`U_values`/`U_ab_values` or `g_values`/`g12_values`,
    `t_values`, `h_values`, `V0_values`, `kL`, `N1_values`, `dmc_shift` --
    exactly as before `model` existed. `coupling_source`, `L_values` and
    the native spin-coupling lists are not used (must be left at their
    defaults/empty).
  - `MODEL_XXZ` / `MODEL_TFIM`: the effective spin-1/2 XXZ ring in
    transverse+longitudinal field (TFIM is the same Hamiltonian with the
    sigma+sigma- "XY" term switched off, chosen automatically by `model`).
    Governed by `coupling_source`:
      - `COUPLING_RBH` (default): derive the spin couplings from
        Rabi-Bose-Hubbard parameters via second-order degenerate
        perturbation theory (`XXZModel.rbh_to_xxz_params`), using the same
        `size_pairs`/`parametrization`/`J_values`.../`t_values`/`h_values`/
        `V0_values` machinery as `MODEL_RABI` (unit filling `N == L` is
        required and enforced by `rbh_to_xxz_params` itself). `N1_values`
        and `dmc_shift` are not applicable to spin models and must be
        left empty/`false`.
      - `COUPLING_NATIVE`: give the spin couplings directly via
        `L_values` (ring sizes, replacing `size_pairs`), `Jz_values`
        (required), and optional `Jxy_values`/`hx_values`/`hz_values`
        (each defaulting to `[0.0]` if left empty). None of
        `size_pairs`/`t_values`/`h_values`/`J_values`/`U_values`/
        `U_ab_values`/`g_values`/`g12_values`/`V0_values` are used in this
        mode (`h_values` must be left at its default `[0.0]`; the other
        unused lists must be empty).

# Interaction parametrization (`parametrization`, for `MODEL_RABI` and
  `MODEL_XXZ`/`MODEL_TFIM` with `coupling_source = COUPLING_RBH`)

The model always assumes a-b (intra-species) symmetry, so there is a
single interaction strength per site (`U` in `PARAM_LATTICE`, `g` in
`PARAM_MICROSCOPIC`) rather than independent `U_a`/`U_b` or `g_up`/`g_down`.

`V0_values` is always given in recoil units (`V0 / E_r`); for `MODEL_RABI`
it is converted to the absolute energy `V0_energy = V0 * kL^2 / 2` that
`params.dat` stores and that the ED/DMRG solvers' `mathieu_band_edges`/
`dmc_shift` step actually consumes. For spin models it is only used (and
only written to `params.dat`) when `parametrization = PARAM_MICROSCOPIC`,
since that is the only case where it enters the physics (via the Wannier
calculation below); with `PARAM_LATTICE` it has no effect on a spin
model's folders, so prefer a single-element `V0_values` there to avoid
generating physically-identical duplicate folders.

  - `PARAM_LATTICE` (default): provide `J_values`, `U_values` and
    `U_ab_values` directly; `kL` (default `pi`) is used by `MODEL_RABI` to
    convert `V0_values` to the absolute `V0_energy`.
  - `PARAM_MICROSCOPIC`: provide `g_values` and `g12_values` instead;
    `J`, `U = g * onsite_overlap` and `U_ab = g12 * onsite_overlap` are
    derived from `V0_values` (and the ring geometry) via
    `compute_lattice_parameters`, assuming unit lattice spacing (continuum
    box length `L_box = L`, so `kL = pi` is derived, not taken from the
    `kL` keyword). Since `J` and the onsite overlap depend only on
    `(V0, L)` and not on `g`/`g12`, the Wannier calculation is only done
    once per `(N, L, V0)` combination and then reused for every
    `(g, g12)` pair.

Every combination is validated through the corresponding solver's own
parameter type (`RabiModel.RabiParams` for `MODEL_RABI`,
`XXZModel.SpinParams` for spin models, and `XXZModel.rbh_to_xxz_params`'s
own checks -- including unit filling `N == L` -- for `coupling_source =
COUPLING_RBH`), i.e. the exact same checks the solvers themselves apply,
so an invalid combination fails here instead of silently creating an
unrunnable folder.

`nsweeps`/`maxdims`/`cutoff` are DMRG-only numerical settings (ignored,
and not written to params.dat, when `solver == SOLVER_ED`).

Returns the number of folders created.
"""
function generate_folders(;
    model::String=MODEL_RABI,
    solver::String,
    parametrization::String=PARAM_LATTICE,
    coupling_source::String=COUPLING_RBH,
    size_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
    L_values::Vector{Int}=Int[],
    t_values::Vector{Float64}=Float64[],
    h_values::Vector{Float64}=[0.0],
    V0_values::Vector{Float64}=[0.0],
    kL::Float64=Float64(π),
    J_values::Vector{Float64}=Float64[],
    U_values::Vector{Float64}=Float64[],
    U_ab_values::Vector{Float64}=Float64[],
    g_values::Vector{Float64}=Float64[],
    g12_values::Vector{Float64}=Float64[],
    Jxy_values::Vector{Float64}=Float64[],
    Jz_values::Vector{Float64}=Float64[],
    hx_values::Vector{Float64}=Float64[],
    hz_values::Vector{Float64}=Float64[],
    wannier_cutoff::Int=12,
    wannier_points_per_site::Int=500,
    N1_values::Vector{Int}=Int[],
    dmc_shift::Bool=false,
    prefix::String="run",
    parent_folder::String="",
    nsweeps::Int=10,
    maxdims::Vector{Int}=[10, 20, 50, 100, 200],
    cutoff::Float64=1e-10,
)
    model in MODELS || throw(ArgumentError("model must be one of $(join(MODELS, ", "))"))
    solver in SOLVERS || throw(ArgumentError("solver must be one of $(join(SOLVERS, ", "))"))
    is_rabi = model == MODEL_RABI
    is_native = !is_rabi && coupling_source == COUPLING_NATIVE

    if is_rabi
        coupling_source == COUPLING_RBH ||
            throw(ArgumentError("coupling_source is only used for model=$MODEL_XXZ/$MODEL_TFIM"))
        isempty(L_values) ||
            throw(ArgumentError("L_values is only used for model=$MODEL_XXZ/$MODEL_TFIM with coupling_source=$COUPLING_NATIVE"))
        (isempty(Jxy_values) && isempty(Jz_values) && isempty(hx_values) && isempty(hz_values)) ||
            throw(ArgumentError("Jxy_values/Jz_values/hx_values/hz_values need model=$MODEL_XXZ/$MODEL_TFIM with coupling_source=$COUPLING_NATIVE"))
    else
        coupling_source in COUPLING_SOURCES ||
            throw(ArgumentError("coupling_source must be one of $(join(COUPLING_SOURCES, ", "))"))
        isempty(N1_values) ||
            throw(ArgumentError("N1_values is only supported by model=$MODEL_RABI"))
        !dmc_shift || throw(ArgumentError("dmc_shift is only meaningful for model=$MODEL_RABI"))
    end

    if is_native
        !isempty(L_values) || throw(ArgumentError("coupling_source=$COUPLING_NATIVE needs a nonempty L_values"))
        all(>=(2), L_values) || throw(ArgumentError("L_values entries must be >= 2"))
        !isempty(Jz_values) || throw(ArgumentError("coupling_source=$COUPLING_NATIVE needs a nonempty Jz_values"))
        isempty(size_pairs) ||
            throw(ArgumentError("size_pairs is not used by coupling_source=$COUPLING_NATIVE; use L_values instead"))
        for (name, vals) in (
            ("t_values", t_values), ("J_values", J_values), ("U_values", U_values),
            ("U_ab_values", U_ab_values), ("g_values", g_values), ("g12_values", g12_values),
        )
            isempty(vals) || throw(ArgumentError("$name is not used by coupling_source=$COUPLING_NATIVE"))
        end
        h_values == [0.0] ||
            throw(ArgumentError("h_values is not used by coupling_source=$COUPLING_NATIVE; use hz_values instead"))
    else
        !isempty(size_pairs) && all(p -> p[1] > 0 && p[2] > 0, size_pairs) ||
            throw(ArgumentError("size_pairs must be a nonempty list of (N, L) with positive entries"))
        isempty(L_values) ||
            throw(ArgumentError("L_values is only used by coupling_source=$COUPLING_NATIVE; use size_pairs instead"))
        for (name, vals) in (("t_values", t_values), ("h_values", h_values), ("V0_values", V0_values))
            !isempty(vals) || throw(ArgumentError("$name must be a nonempty list"))
        end
        parametrization in PARAMETRIZATIONS ||
            throw(ArgumentError("parametrization must be one of $(join(PARAMETRIZATIONS, ", "))"))
        if parametrization == PARAM_LATTICE
            (!isempty(J_values) && !isempty(U_values) && !isempty(U_ab_values)) ||
                throw(ArgumentError("parametrization=$PARAM_LATTICE needs nonempty J_values, U_values and U_ab_values"))
            (isempty(g_values) && isempty(g12_values)) ||
                throw(ArgumentError("g_values/g12_values need parametrization=$PARAM_MICROSCOPIC"))
        else
            (!isempty(g_values) && !isempty(g12_values)) ||
                throw(ArgumentError("parametrization=$PARAM_MICROSCOPIC needs nonempty g_values and g12_values"))
            (isempty(J_values) && isempty(U_values) && isempty(U_ab_values)) ||
                throw(ArgumentError("J_values/U_values/U_ab_values need parametrization=$PARAM_LATTICE"))
            (all(>=(0), g_values) && all(>=(0), g12_values)) ||
                throw(ArgumentError("g_values/g12_values must be nonnegative (repulsive)"))
        end
    end

    if is_rabi && !isempty(N1_values)
        solver == SOLVER_ED ||
            throw(ArgumentError("N1_values is only supported by solver=$SOLVER_ED"))
        all(==(0), t_values) ||
            throw(ArgumentError("N1_values can only be used when all t_values == 0"))
    end

    count = 0

    if is_native
        Jxy_variants = isempty(Jxy_values) ? [0.0] : Jxy_values
        hx_variants = isempty(hx_values) ? [0.0] : hx_values
        hz_variants = isempty(hz_values) ? [0.0] : hz_values
        for L in L_values, Jxy in Jxy_variants, Jz in Jz_values, hx in hx_variants, hz in hz_variants
            SpinParams(; L, Jxy, Jz, hx, hz)
            _write_spin_native_folder(;
                parent_folder, prefix, model, solver, L, Jxy, Jz, hx, hz,
                nsweeps, maxdims, cutoff,
            )
            count += 1
        end
        return count
    end

    N1_variants = isempty(N1_values) ? [nothing] : N1_values
    lattice_combos = _lattice_param_combos(;
        parametrization, size_pairs, J_values, U_values, U_ab_values,
        g_values, g12_values, V0_values, kL, wannier_cutoff, wannier_points_per_site,
    )

    for combo in lattice_combos, t in t_values, h in h_values
        N, L, J, U, U_ab = combo.N, combo.L, combo.J, combo.U, combo.U_ab
        if is_rabi
            for N1 in N1_variants
                V0_energy = combo.V0_recoil * combo.kL^2 / 2
                _validate_combo(N, N1)
                RabiParams(; N, L, J, U_a=U, U_b=U, U_ab, t, h, V0=V0_energy, kL=combo.kL)
                _write_rabi_folder(;
                    parent_folder, prefix, solver, N, L, J, U_a=U, U_b=U, U_ab, t, h,
                    V0=V0_energy, V0_recoil=combo.V0_recoil, kL=combo.kL, N1, dmc_shift,
                    nsweeps, maxdims, cutoff, extra_fields=combo.extra_fields,
                )
                count += 1
            end
        else
            include_xy = model == MODEL_XXZ
            sp, C = rbh_to_xxz_params(N, L, J, U, U_ab, t, h)
            _write_spin_rbh_folder(;
                parent_folder, prefix, model, solver, N, L, J, U_a=U, U_b=U, U_ab, t, h,
                Jxy=sp.Jxy, Jz=sp.Jz, hx=sp.hx, hz=sp.hz, C,
                V0_recoil=combo.V0_recoil, kL=combo.kL,
                nsweeps, maxdims, cutoff, extra_fields=combo.extra_fields,
            )
            count += 1
        end
    end

    return count
end

# Shared lattice/microscopic derivation of (J, U, U_ab), reused by both
# model=rabi and model=xxz/tfim with coupling_source=rbh.
function _lattice_param_combos(;
    parametrization::String, size_pairs::Vector{Tuple{Int,Int}},
    J_values::Vector{Float64}, U_values::Vector{Float64}, U_ab_values::Vector{Float64},
    g_values::Vector{Float64}, g12_values::Vector{Float64},
    V0_values::Vector{Float64}, kL::Float64,
    wannier_cutoff::Int, wannier_points_per_site::Int,
)
    combos = NamedTuple[]
    if parametrization == PARAM_LATTICE
        for (N, L) in size_pairs, J in J_values, U in U_values, U_ab in U_ab_values, V0 in V0_values
            push!(combos, (; N, L, J, U, U_ab, V0_recoil=V0, kL=kL, extra_fields=Pair{String,Float64}[]))
        end
    else
        for (N, L) in size_pairs, V0 in V0_values
            # J and the onsite overlap depend only on (V0, L), so the
            # (expensive) Wannier calculation is cached here and reused
            # for every (g, g12) combination below.
            L_box = Float64(L)
            lattice = compute_lattice_parameters(;
                L=L_box, lattice_sites=L, V0=V0, g_up=1.0, g_down=1.0, g_updown=1.0,
                cutoff=wannier_cutoff, points_per_site=wannier_points_per_site,
            )
            J = lattice.J
            overlap = lattice.onsite_overlap
            kL_derived = π * L / L_box
            for g in g_values, g12 in g12_values
                U = g * overlap
                U_ab = g12 * overlap
                push!(combos, (; N, L, J, U, U_ab, V0_recoil=V0, kL=kL_derived, extra_fields=["g" => g, "g12" => g12]))
            end
        end
    end
    return combos
end

function _validate_combo(N::Int, N1::Union{Nothing,Int})
    N1 === nothing || (0 <= N1 <= N || throw(ArgumentError("N1=$N1 must be between 0 and N=$N")))
    return nothing
end

function _create_folder(parent_folder::String, folder_name::String)
    folder = isempty(parent_folder) ? folder_name : joinpath(parent_folder, folder_name)
    mkpath(folder)
    return folder
end

function _write_rabi_folder(;
    parent_folder::String, prefix::String, solver::String,
    N::Int, L::Int, J::Real, U_a::Real, U_b::Real, U_ab::Real,
    t::Real, h::Real, V0::Real, V0_recoil::Real, kL::Real,
    N1::Union{Nothing,Int}, dmc_shift::Bool,
    nsweeps::Int, maxdims::Vector{Int}, cutoff::Real,
    extra_fields::Vector{Pair{String,Float64}}=Pair{String,Float64}[],
)
    n1_tag = N1 === nothing ? "" : @sprintf("_N1_%d", N1)
    folder_name = @sprintf(
        "%s_%s_%s_N%d_L%d_J%.6f_U%.6f_Uab%.6f_t%.6f_h%.6f_V0%.6f",
        prefix, MODEL_RABI, solver, N, L, J, U_a, U_ab, t, h, V0_recoil,
    ) * n1_tag
    folder = _create_folder(parent_folder, folder_name)

    open(joinpath(folder, "params.dat"), "w") do f
        println(f, "model               $MODEL_RABI")
        println(f, "solver              $solver")
        println(f, "N                   $N")
        println(f, "L                   $L")
        @printf(f, "J                   %.12f\n", J)
        @printf(f, "U_a                 %.12f\n", U_a)
        @printf(f, "U_b                 %.12f\n", U_b)
        @printf(f, "U_ab                %.12f\n", U_ab)
        @printf(f, "t                   %.12f\n", t)
        @printf(f, "h                   %.12f\n", h)
        @printf(f, "V0                  %.12f\n", V0)
        @printf(f, "kL                  %.12f\n", kL)
        println(f, "dmc_shift           $dmc_shift")
        N1 === nothing || println(f, "N1                  $N1")
        for (key, val) in extra_fields
            @printf(f, "%-20s%.12f\n", key, val)
        end
        if solver == SOLVER_DMRG
            println(f, "nsweeps             $nsweeps")
            println(f, "maxdims             $(join(maxdims, ","))")
            @printf(f, "cutoff              %.3e\n", cutoff)
        end
    end

    println("Created: $folder")
    return nothing
end

function _write_spin_rbh_folder(;
    parent_folder::String, prefix::String, model::String, solver::String,
    N::Int, L::Int, J::Real, U_a::Real, U_b::Real, U_ab::Real, t::Real, h::Real,
    Jxy::Real, Jz::Real, hx::Real, hz::Real, C::Real,
    V0_recoil::Real, kL::Real,
    nsweeps::Int, maxdims::Vector{Int}, cutoff::Real,
    extra_fields::Vector{Pair{String,Float64}}=Pair{String,Float64}[],
)
    folder_name = @sprintf(
        "%s_%s_%s_N%d_L%d_J%.6f_U%.6f_Uab%.6f_t%.6f_h%.6f",
        prefix, model, solver, N, L, J, U_a, U_ab, t, h,
    )
    folder = _create_folder(parent_folder, folder_name)

    open(joinpath(folder, "params.dat"), "w") do f
        println(f, "model               $model")
        println(f, "coupling_source     $COUPLING_RBH")
        println(f, "solver              $solver")
        println(f, "N                   $N")
        println(f, "L                   $L")
        @printf(f, "J                   %.12f\n", J)
        @printf(f, "U_a                 %.12f\n", U_a)
        @printf(f, "U_b                 %.12f\n", U_b)
        @printf(f, "U_ab                %.12f\n", U_ab)
        @printf(f, "t                   %.12f\n", t)
        @printf(f, "h                   %.12f\n", h)
        @printf(f, "Jxy                 %.12f\n", Jxy)
        @printf(f, "Jz                  %.12f\n", Jz)
        @printf(f, "hx                  %.12f\n", hx)
        @printf(f, "hz                  %.12f\n", hz)
        @printf(f, "C                   %.12f\n", C)
        if !isempty(extra_fields)
            @printf(f, "V0                  %.12f\n", V0_recoil)
            @printf(f, "kL                  %.12f\n", kL)
            for (key, val) in extra_fields
                @printf(f, "%-20s%.12f\n", key, val)
            end
        end
        if solver == SOLVER_DMRG
            println(f, "nsweeps             $nsweeps")
            println(f, "maxdims             $(join(maxdims, ","))")
            @printf(f, "cutoff              %.3e\n", cutoff)
        end
    end

    println("Created: $folder")
    return nothing
end

function _write_spin_native_folder(;
    parent_folder::String, prefix::String, model::String, solver::String,
    L::Int, Jxy::Real, Jz::Real, hx::Real, hz::Real,
    nsweeps::Int, maxdims::Vector{Int}, cutoff::Real,
)
    folder_name = @sprintf(
        "%s_%s_%s_L%d_Jxy%.6f_Jz%.6f_hx%.6f_hz%.6f",
        prefix, model, solver, L, Jxy, Jz, hx, hz,
    )
    folder = _create_folder(parent_folder, folder_name)

    open(joinpath(folder, "params.dat"), "w") do f
        println(f, "model               $model")
        println(f, "coupling_source     $COUPLING_NATIVE")
        println(f, "solver              $solver")
        println(f, "L                   $L")
        @printf(f, "Jxy                 %.12f\n", Jxy)
        @printf(f, "Jz                  %.12f\n", Jz)
        @printf(f, "hx                  %.12f\n", hx)
        @printf(f, "hz                  %.12f\n", hz)
        if solver == SOLVER_DMRG
            println(f, "nsweeps             $nsweeps")
            println(f, "maxdims             $(join(maxdims, ","))")
            @printf(f, "cutoff              %.3e\n", cutoff)
        end
    end

    println("Created: $folder")
    return nothing
end
