using Pkg
Pkg.add(["Plots", "Zygote", "Einsum", "DelimitedFiles"])

using Base: @kwdef
using LinearAlgebra
using Plots
using Einsum
using Zygote
using DelimitedFiles

@kwdef struct SimulationConstants
    m::Float64 = 39.9748     # g/mol, mass
    T::Float64 = 300.0       # K, temperature
    sigma::Float64 = 3.4     # A, particle size
    eps::Float64 = 0.24      # kcal/mol interactions_strength
    # Box defined by 2 corners (xi,yi), (xf,yf)
    xi::Float64 = 0.0        # A
    yi::Float64 = 0.0        # A
    xf::Float64 = 20.0       # A
    yf::Float64 = 20.0       # A
    a_unit_factor_kcal_per_gA_to_A_per_ps2::Float64 = 418.4  # A/(ps^2)/ (kcal/(g A))
    kB::Float64 = 1.381e-23  # J/K
    kin_to_j::Float64 = 1.661e-23  # J/( g * Å^2 / (ps^2 * mol) )
    kin_to_pot::Float64 = 0.00239  # g *Å^2/(ps^2 * mol) to kcal/mol
end


abstract type MDSimulation end

mutable struct MDSimulationBase{T<:AbstractMatrix} <: MDSimulation
    positions::T
    velocities::T
    accelerations::T
    constants::SimulationConstants
    total_steps::Int
    total_reflections::Int
end

function MDSimulationBase(constants::SimulationConstants, pos_file::String="initial_pos.csv", vel_file::String="initial_vel.csv")
    positions = Matrix{Float64}(readdlm(pos_file))
    velocities = Matrix{Float64}(readdlm(vel_file))

    if size(positions) != size(velocities)
        throw(ArgumentError("Positions and velocities must have the same shape"))
    end

    accelerations = zeros(size(positions))


    sim = MDSimulationBase(positions, velocities, accelerations, constants, 0, 0)

    sim.accelerations = get_acceleration(sim)
    return sim
end

function get_kinetic_energy(sim::MDSimulation; debug=false)
    v = sim.velocities
    @einsum v2[p] := v[p, c] * v[p, c]
    if debug
        println("v2: \n", v2)
    end
    return 0.5 * sim.constants.m * sum(v2)
end

get_kinetic_energy_mol(sim::MDSimulation) = sim.constants.kin_to_pot * get_kinetic_energy(sim)

function get_instantaneous_temperature(sim::MDSimulation)
    return sim.constants.kin_to_j * 2 * get_kinetic_energy(sim) /
           (sim.constants.kB * prod(size(sim.positions)))
end

function get_force(sim::MDSimulation)
    gradient(positions -> get_potential_energy(sim, positions), sim.positions)[1]
end

function get_acceleration(sim::MDSimulation)
    a = -get_force(sim) / sim.constants.m
    return a * sim.constants.a_unit_factor_kcal_per_gA_to_A_per_ps2
end

function step_reflect!(sim::MDSimulation; debug=false, reflect_log=false)
    corners = [
        [sim.constants.xi sim.constants.yi];
        [sim.constants.xf sim.constants.yf]
    ]

    # Calculate differences to both corners
    diff_to_axis = zeros(2, size(sim.positions)...)
    for i in 1:size(sim.positions, 1)
        for j in 1:size(sim.positions, 2)
            diff_to_axis[1, i, j] = sim.positions[i, j] - corners[1, j]
            diff_to_axis[2, i, j] = sim.positions[i, j] - corners[2, j]
        end
    end

    # Apply constraints
    diff_to_axis[1, :, :] = min.(diff_to_axis[1, :, :], 0)
    diff_to_axis[2, :, :] = max.(diff_to_axis[2, :, :], 0)

    # Calculate position adjustments
    @einsum pos_adj[p, c] := -2 * diff_to_axis[h, p, c]

    # Calculate velocity adjustments
    vel_adj = ifelse.(abs.(pos_adj) .> 0, -1, 1)

    # Count reflections
    reflections = count(x -> x > 0, abs.(pos_adj))
    sim.total_reflections += reflections

    if reflect_log && reflections > 0
        println("amount of reflections: ", reflections)
    end

    if debug
        println("Debug Information: Reflect Step")
        println("-"^40)
        println("Current Boundary Corners:")
        println("Bottom Left Corner: ($(sim.constants.xi), $(sim.constants.yi))")
        println("Top Right Corner: ($(sim.constants.xf), $(sim.constants.yf))")
        println("\nPosition Adjustments:")
        println(pos_adj)
        println("\nVelocity Adjustments:")
        println(vel_adj)
        println("-"^40)
        plot_points(sim)
    end

    sim.positions .+= pos_adj
    sim.velocities .*= vel_adj

    if debug
        plot_points(sim)
    end
end

function plot_points(sim::MDSimulation)
    p = Plots.scatter(sim.positions[:, 1], sim.positions[:, 2],
        label="Particles", color=:blue, aspect_ratio=:equal)

    # Annotate points
    for i in 1:size(sim.positions, 1)
        annotate!(p, [(sim.positions[i, 1], sim.positions[i, 2], "$i")])

        # Velocity arrows
        quiver!(p, [sim.positions[i, 1]], [sim.positions[i, 2]],
            quiver=([sim.velocities[i, 1]], [sim.velocities[i, 2]]),
            color=:red)

        # Acceleration arrows
        quiver!(p, [sim.positions[i, 1]], [sim.positions[i, 2]],
            quiver=([sim.accelerations[i, 1]], [sim.accelerations[i, 2]]),
            color=:green)
    end

    # Draw box
    box_x = [sim.constants.xi, sim.constants.xf, sim.constants.xf, sim.constants.xi, sim.constants.xi]
    box_y = [sim.constants.yi, sim.constants.yi, sim.constants.yf, sim.constants.yf, sim.constants.yi]
    plot!(p, box_x, box_y, color=:black, linestyle=:dash, label="Box")

    # Add dummy plots for legend
    plot!(p, [], [], color=:red, label="Velocities")
    plot!(p, [], [], color=:green, label="Accelerations")

    xlabel!(p, "x-axis")
    ylabel!(p, "y-axis")
    xlims!(p, sim.constants.xi - 5, sim.constants.xf + 5)
    ylims!(p, sim.constants.yi - 5, sim.constants.yf + 5)
    title!(p, "Particle Positions with Velocities and Accelerations")

    display(p)
end


using StaticArrays

function get_potential_energy(sim::MDSimulationBase, positions=nothing; debug=false)
    pos = isnothing(positions) ? sim.positions : positions
    n = size(pos, 1)
    σ = sim.constants.sigma
    ϵ = sim.constants.eps

    # Use static array for position differences to avoid allocations
    dims = size(pos, 2)
    diff = @SVector zeros(dims)

    # Accumulate potential
    total_potential = zero(eltype(pos))

    @inbounds for i in 1:n
        pi = @view pos[i, :]
        for j in (i+1):n
            pj = @view pos[j, :]
            # Calculate r² directly
            r² = zero(eltype(pos))
            @simd for k in 1:dims
                δr = pi[k] - pj[k]
                r² += δr * δr
            end

            # Inverse operations to avoid division
            inv_r = 1 / sqrt(r²)
            inv_r6 = (σ * inv_r)^6

            # Final potential calculation
            total_potential += 4ϵ * inv_r6 * (inv_r6 - 1)
        end
    end

    return total_potential
end


function step!(sim::MDSimulationBase, dt::Float64; reflect_log=false)
    sim.positions .+= sim.velocities .* dt .+ 0.5 .* sim.accelerations .* dt^2
    step_reflect!(sim, reflect_log=reflect_log)
    new_a = get_acceleration(sim)
    sim.velocities .+= 0.5 .* (sim.accelerations .+ new_a) .* dt
    sim.accelerations = new_a
    sim.total_steps += 1
end
