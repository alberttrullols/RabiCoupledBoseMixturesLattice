include(joinpath(@__DIR__, "constants.jl"))

function read_params(filename::String)
    params = Dict{String,String}()
    open(filename) do file
        for (line_number, line) in enumerate(eachline(file))
            line = strip(line)
            isempty(line) && continue
            startswith(line, "#") && continue
            fields = split(line)
            length(fields) >= 2 ||
                throw(ArgumentError("Invalid params.dat entry at line $line_number"))
            params[fields[1]] = fields[2]
        end
    end
    return params
end

function required_param(params::Dict{String,String}, key::String)
    haskey(params, key) || throw(ArgumentError("params.dat is missing required field '$key'"))
    return params[key]
end

param_int(params::Dict{String,String}, key::String) =
    parse(Int, required_param(params, key))
param_float(params::Dict{String,String}, key::String) =
    parse(Float64, required_param(params, key))
param_bool(params::Dict{String,String}, key::String) =
    parse(Bool, required_param(params, key))

function optional_int(params::Dict{String,String}, key::String)
    return haskey(params, key) ? parse(Int, params[key]) : nothing
end

function requested_observables(params::Dict{String,String})
    raw = get(params, "observables", string(OBS_ENERGY))
    requested = raw == "none" ? Symbol[] : Symbol.(split(raw, ","))
    unknown = setdiff(requested, AVAILABLE_OBSERVABLES)
    isempty(unknown) || throw(ArgumentError("Unknown observables: $(join(unknown, ", "))"))
    return unique(requested)
end

function validate_observables(solver::String, requested::Vector{Symbol})
    solver == SOLVER_IDMRG || return nothing
    supported = (OBS_ENERGY, OBS_ENERGY_PER_PARTICLE)
    unsupported = filter(observable -> !(observable in supported), requested)
    isempty(unsupported) ||
        throw(ArgumentError(
            "Observables $(join(unsupported, ", ")) are not implemented for iDMRG states"
        ))
    return nothing
end

function solver_settings(params::Dict{String,String})
    maxdims = parse.(Int, split(get(params, "maxdims", "10,20,50,100,200"), ","))
    return (;
        nsweeps=parse(Int, get(params, "nsweeps", "10")),
        maxdims=maxdims,
        cutoff=parse(Float64, get(params, "cutoff", "1e-10")),
    )
end

function load_solver(model::String, solver::String)
    if model == MODEL_RABI
        solver == SOLVER_ED && return include(joinpath(@__DIR__, "src", "rabi_ed.jl"))
        solver == SOLVER_DMRG && return include(joinpath(@__DIR__, "src", "rabi_dmrg.jl"))
        solver == SOLVER_IDMRG && return include(joinpath(@__DIR__, "src", "rabi_idmrg.jl"))
    else
        filename = solver == SOLVER_ED ? "xxz_ed.jl" :
                   solver == SOLVER_DMRG ? "xxz_dmrg.jl" :
                   solver == SOLVER_IDMRG ? "xxz_idmrg.jl" : ""
        isempty(filename) || return include(joinpath(@__DIR__, "src", filename))
    end
    throw(ArgumentError("Unsupported model=$model solver=$solver"))
end

function run_rabi(params::Dict{String,String}, solver::String)
    N, L = param_int(params, "N"), param_int(params, "L")
    J, U_a = param_float(params, "J"), param_float(params, "U_a")
    U_b, U_ab = param_float(params, "U_b"), param_float(params, "U_ab")
    t, h = param_float(params, "t"), param_float(params, "h")
    V0, kL = param_float(params, "V0"), param_float(params, "kL")
    dmc_shift = param_bool(params, "dmc_shift")
    model_params = RabiParams(; N, L, J, U_a, U_b, U_ab, t, h, V0, kL)

    if solver == SOLVER_ED
        energy, psi, basis = two_component_ground_state(
            N, L, J, U_a, U_b, U_ab, t, h, V0, kL;
            N1=optional_int(params, "N1"), dmc_shift,
        )
        return (; energy, energy_kind="total", psi, basis, sites=nothing, model_params)
    end

    settings = solver_settings(params)
    if solver == SOLVER_DMRG
        energy, psi, sites, model_params = two_component_dmrg_ground_state(
            N, L, J, U_a, U_b, U_ab, t, h, V0, kL;
            dmc_shift, settings...,
        )
        return (; energy, energy_kind="total", psi, basis=nothing, sites, model_params)
    end

    energy, psi, sites, model_params = two_component_idmrg_ground_state(
        model_params;
        dmc_shift,
        unitcell_size=parse(Int, get(params, "unitcell_size", "2")),
        max_occupancy=parse(Int, get(params, "max_occupancy", "2")),
        nsteps_per_stage=parse(Int, get(params, "nsteps_per_stage", "3")),
        settings...,
    )
    return (; energy, energy_kind="per_site", psi, basis=nothing, sites, model_params)
end

function run_spin(params::Dict{String,String}, model::String, solver::String)
    coupling_source = get(params, "coupling_source", COUPLING_RBH)
    include_xy = model != MODEL_TFIM

    coupling_source in (COUPLING_NATIVE, COUPLING_RBH) ||
        throw(ArgumentError("Unsupported coupling_source=$coupling_source"))
    spin_params = SpinParams(;
        L=param_int(params, "L"), Jxy=param_float(params, "Jxy"),
        Jz=param_float(params, "Jz"), hx=param_float(params, "hx"),
        hz=param_float(params, "hz"),
    )

    settings = solver_settings(params)
    energy, psi, sites = if solver == SOLVER_ED
        energy, psi = spin_ground_state(spin_params; include_xy)
        energy, psi, nothing
    elseif solver == SOLVER_DMRG
        spin_dmrg_ground_state(spin_params; include_xy, settings...)
    else
        energy, psi, sites, _ = spin_idmrg_ground_state(
            spin_params; include_xy,
            unitcell_size=parse(Int, get(params, "unitcell_size", string(spin_params.L))),
            nsteps_per_stage=parse(Int, get(params, "nsteps_per_stage", "3")),
            settings...,
        )
        energy, psi, sites
    end

    if coupling_source == COUPLING_RBH
        C = param_float(params, "C")
        energy += solver == SOLVER_IDMRG ? C : spin_params.L * C
    end
    energy_kind = solver == SOLVER_IDMRG ? "per_site" : "total"
    return (; energy, energy_kind, psi, sites, model_params=spin_params)
end

function calculate_observables(
    requested::Vector{Symbol}, result, model::String, solver::String,
)
    values = Dict{Symbol,Any}()
    for observable in requested
        if observable == OBS_ENERGY
            values[observable] = result.energy
        elseif observable == OBS_ENERGY_PER_PARTICLE
            count = model == MODEL_RABI ? result.model_params.N : result.model_params.L
            values[observable] = result.energy_kind == "per_site" ?
                result.energy : energy_per_particle(result.energy, count)
        elseif model == MODEL_RABI
            psi, params = result.psi, result.model_params
            is_ed = solver == SOLVER_ED
            basis, sites = result.basis, result.sites
            if observable == OBS_P
                values[observable] = is_ed ?
                    polarization(psi, basis, params.N, params.L) : polarization(psi, params)
            elseif observable == OBS_P2
                values[observable] = is_ed ?
                    square_polarization(psi, basis, params.N, params.L) :
                    square_polarization(psi, params)
            elseif observable == OBS_P4
                values[observable] = is_ed ?
                    quartic_polarization(psi, basis, params.N, params.L) :
                    quartic_polarization(psi, params, sites)
            elseif observable == OBS_PX
                values[observable] = is_ed ?
                    transverse_polarization(psi, basis, params.N, params.L) :
                    transverse_polarization(psi, params)
            else
                D = is_ed ?
                    polarization_correlation_matrix(psi, basis, params.L) :
                    polarization_correlation_matrix(psi, params)
                values[observable] = if observable == OBS_POLARIZATION_CORRELATIONS
                    D
                elseif observable == OBS_SPIN_SPIN_CORRELATIONS
                    spin_spin_correlations(D)
                elseif observable == OBS_SK
                    structure_factor(D, 0)
                elseif observable == OBS_SK_SPECTRUM
                    ks, sk = structure_factor_spectrum(D)
                    (; k=ks, values=sk)
                else
                    throw(ArgumentError("Observable $observable is not implemented for model=$model"))
                end
            end
        else
            psi, L = result.psi, result.model_params.L
            is_ed = solver == SOLVER_ED
            if observable == OBS_P
                values[observable] = polarization(psi, L)
            elseif observable == OBS_P2
                values[observable] = square_polarization(psi, L)
            elseif observable == OBS_P4
                values[observable] = is_ed ?
                    quartic_polarization(psi, L) : quartic_polarization(psi, L, result.sites)
            elseif observable == OBS_PX
                values[observable] = transverse_polarization(psi, L)
            else
                D = polarization_correlation_matrix(psi, L)
                values[observable] = if observable == OBS_POLARIZATION_CORRELATIONS
                    D
                elseif observable == OBS_SPIN_SPIN_CORRELATIONS
                    spin_spin_correlations(D)
                elseif observable == OBS_SK
                    structure_factor(D, 0)
                elseif observable == OBS_SK_SPECTRUM
                    ks, sk = structure_factor_spectrum(D)
                    (; k=ks, values=sk)
                else
                    throw(ArgumentError("Observable $observable is not implemented for model=$model"))
                end
            end
        end
    end
    return values
end

function write_result_line(io::IO, name::AbstractString, value)
    println(io, name, repeat(" ", max(1, 36 - length(name))), value)
    return nothing
end

function write_result_value(io::IO, name::Symbol, value::Number)
    write_result_line(io, string(name), value)
    return nothing
end

function write_result_value(io::IO, name::Symbol, value::AbstractVector)
    write_result_line(io, "$(name)_values", join(value, ","))
    return nothing
end

function write_result_value(io::IO, name::Symbol, value::AbstractMatrix)
    rows, columns = size(value)
    write_result_line(io, "$(name)_shape", "$rows,$columns")
    values = (value[i, j] for i in 1:rows for j in 1:columns)
    write_result_line(io, "$(name)_values", join(values, ","))
    return nothing
end

function write_result_value(io::IO, name::Symbol, value::NamedTuple)
    hasproperty(value, :k) && hasproperty(value, :values) ||
        throw(ArgumentError("Unsupported named-tuple observable result for $name"))
    write_result_line(io, "$(name)_k", join(value.k, ","))
    write_result_line(io, "$(name)_values", join(value.values, ","))
    return nothing
end

function write_results(
    params_file::String, model::String, solver::String, result,
    requested::Vector{Symbol}, values::Dict{Symbol,Any},
)
    results_file = joinpath(dirname(abspath(params_file)), "results.dat")
    open(results_file, "w") do io
        write_result_line(io, "model", model)
        write_result_line(io, "solver", solver)
        write_result_line(io, "energy_kind", result.energy_kind)
        for observable in requested
            write_result_value(io, observable, values[observable])
        end
    end
    return results_file
end

function main(filename::String="params.dat")
    isfile(filename) || throw(ArgumentError("Parameter file not found: $filename"))
    params = read_params(filename)
    model = required_param(params, "model")
    solver = required_param(params, "solver")
    requested = requested_observables(params)
    model in (MODEL_RABI, MODEL_XXZ, MODEL_TFIM) || throw(ArgumentError("Unsupported model=$model"))
    solver in SOLVERS || throw(ArgumentError("Unsupported solver=$solver"))
    validate_observables(solver, requested)
    load_solver(model, solver)

    # Solver methods and types were just defined by include, in a newer world.
    result = if model == MODEL_RABI
        Base.invokelatest(run_rabi, params, solver)
    else
        Base.invokelatest(run_spin, params, model, solver)
    end
    observables = Base.invokelatest(calculate_observables, requested, result, model, solver)
    write_results(filename, model, solver, result, requested, observables)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || throw(ArgumentError("Usage: julia run_simulation.jl [params.dat]"))
    main(isempty(ARGS) ? "params.dat" : ARGS[1])
end
