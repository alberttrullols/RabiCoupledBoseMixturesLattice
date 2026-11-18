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
    load_solver(MODEL_RABI, solver)
    N, L = param_int(params, "N"), param_int(params, "L")
    J, U_a = param_float(params, "J"), param_float(params, "U_a")
    U_b, U_ab = param_float(params, "U_b"), param_float(params, "U_ab")
    t, h = param_float(params, "t"), param_float(params, "h")
    V0, kL = param_float(params, "V0"), param_float(params, "kL")
    dmc_shift = param_bool(params, "dmc_shift")

    if solver == SOLVER_ED
        energy, _, _ = two_component_ground_state(
            N, L, J, U_a, U_b, U_ab, t, h, V0, kL;
            N1=optional_int(params, "N1"), dmc_shift,
        )
        return energy, "total"
    end

    settings = solver_settings(params)
    if solver == SOLVER_DMRG
        energy, _, _, _ = two_component_dmrg_ground_state(
            N, L, J, U_a, U_b, U_ab, t, h, V0, kL;
            dmc_shift, settings...,
        )
        return energy, "total"
    end

    energy, _, _, _ = two_component_idmrg_ground_state(
        N, L, J, U_a, U_b, U_ab, t, h, V0, kL;
        dmc_shift,
        unitcell_size=parse(Int, get(params, "unitcell_size", "2")),
        max_occupancy=parse(Int, get(params, "max_occupancy", "2")),
        nsteps_per_stage=parse(Int, get(params, "nsteps_per_stage", "3")),
        settings...,
    )
    return energy, "per_site"
end

function run_spin(params::Dict{String,String}, model::String, solver::String)
    load_solver(model, solver)
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
    energy = if solver == SOLVER_ED
        first(spin_ground_state(spin_params; include_xy))
    elseif solver == SOLVER_DMRG
        first(spin_dmrg_ground_state(spin_params; include_xy, settings...))
    else
        first(spin_idmrg_ground_state(
            spin_params; include_xy,
            unitcell_size=parse(Int, get(params, "unitcell_size", string(spin_params.L))),
            nsteps_per_stage=parse(Int, get(params, "nsteps_per_stage", "3")),
            settings...,
        ))
    end

    if coupling_source == COUPLING_RBH
        C = param_float(params, "C")
        energy += solver == SOLVER_IDMRG ? C : spin_params.L * C
    end
    return energy, solver == SOLVER_IDMRG ? "per_site" : "total"
end

function main(filename::String="params.dat")
    isfile(filename) || throw(ArgumentError("Parameter file not found: $filename"))
    params = read_params(filename)
    model = required_param(params, "model")
    solver = required_param(params, "solver")
    energy, energy_kind = if model == MODEL_RABI
        run_rabi(params, solver)
    elseif model == MODEL_XXZ || model == MODEL_TFIM
        run_spin(params, model, solver)
    else
        throw(ArgumentError("Unsupported model=$model"))
    end
    println("model: $model")
    println("solver: $solver")
    println("energy ($energy_kind): $energy")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || throw(ArgumentError("Usage: julia run_simulation.jl [params.dat]"))
    main(isempty(ARGS) ? "params.dat" : ARGS[1])
end
