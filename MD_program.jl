using LinearAlgebra
using Printf
using Random
using PyPlot
using Torch
using StatsBase
using Base.Iterators

# Define a struct to hold simulation constants
struct SimulationConstants
    m::Float64
    T::Float64
    sigma::Float64
    eps::Float64
    xi::Float64
    yi::Float64
    xf::Float64
    yf::Float64
    a_unit_factor_kcal_per_gA_to_A_per_ps2::Float64
    kB::Float64
    kin_to_j::Float64
    kin_to_pot::Float64
end


# Default values for the constants (similar to the Python `dataclass`)
const sim_constants = SimulationConstants(
    39.9748, # g/mol
    300.0,   # K
    3.4,     # A
    0.24,    # kcal/mol
    0.0,     # xi
    0.0,     # yi
    20.0,    # xf
    20.0,    # yf
    418.4,   # conversion factor
    1.381e-23,  # kB
    1.661e-23,  # kin_to_j
    0.00239     # kin_to_pot
)

# Abstract base class for MD simulations
abstract type MDSimulation end

# Define the main simulation struct
mutable struct BaseSimulation <: MDSimulation
    positions::Tensor
    velocities::Tensor
    accelerations::Tensor
    constants::SimulationConstants
    total_steps::Int
    total_reflections::Int

    function BaseSimulation(constants::SimulationConstants, pos_file::String="initial_pos.csv", vel_file::String="initial_vel.csv")
        positions = Tensor(torch_from_numpy(readdlm(pos_file)))
        velocities = Tensor(torch_from_numpy(readdlm(vel_file)))
        accelerations = zeros_like(positions)
        total_steps = 0
        total_reflections = 0

        if size(positions) != size(velocities)
            error("Positions and velocities must have the same shape")
        end

        new(constants, positions, velocities, accelerations, total_steps, total_reflections)
    end
end

# Methods for BaseSimulation
function get_kinetic_energy(sim::BaseSimulation; debug=false)
    v2 = torch.einsum("p c, p c -> p", sim.velocities, sim.velocities)
    if debug
        println("v2: ", v2.detach().numpy())
    end
    return 0.5 * sim.constants.m * torch.einsum("p ->", v2)
end

function get_kinetic_energy_mol(sim::BaseSimulation)
    return sim.constants.kin_to_pot * get_kinetic_energy(sim)
end

function get_instantaneous_temperature(sim::BaseSimulation)
    return sim.constants.kin_to_j * 2 * get_kinetic_energy(sim) / (sim.constants.kB * prod(size(sim.positions)))
end

function get_force(sim::BaseSimulation)
    positions = sim.positions.detach().requires_grad_(true)
    energy = get_potential_energy(sim, positions)
    forces = torch.autograd.grad(energy, positions)[1]
    return -forces
end

function get_acceleration(sim::BaseSimulation)
    a = get_force(sim) / sim.constants.m
    return a * sim.constants.a_unit_factor_kcal_per_gA_to_A_per_ps2
end

# Define a reflection function
function step_reflect!(sim::BaseSimulation; debug=false, reflect_log=false)
    diff_to_axis = rearrange(sim.positions, "p c -> 1 p c") .- torch.tensor([
        [sim.constants.xi sim.constants.yi],
        [sim.constants.xf sim.constants.yf]
    ])

    diff_to_axis[1, :, :] = clamp.(diff_to_axis[1, :, :], -Inf, 0)
    pos_adj = -2 * torch.einsum("h p c -> p c", diff_to_axis)
    vel_adj = torch.where(torch.abs(pos_adj) .> 0, -1, 1)

    adj = torch.einsum("p c -> p", abs.(pos_adj))
    amount_reflections = torch.einsum("p->", torch.where(adj .> 0, 1, 0))

    sim.total_reflections += amount_reflections
    if reflect_log && amount_reflections > 0
        println("Amount of reflections: ", amount_reflections)
    end

    sim.positions .+= pos_adj
    sim.velocities .= sim.velocities .* vel_adj

    if debug
        plt_points(sim)
    end
end

function plt_points(sim::BaseSimulation)
    pos = sim.positions.detach().numpy()
    vel = sim.velocities.detach().numpy()
    acc = sim.accelerations.detach().numpy()

    scatter(pos[:, 1], pos[:, 2], color="blue", label="Particles")
    for i in 1:size(pos, 1)
        x, y = pos[i, 1], pos[i, 2]
        annotate(string(i), (x, y), textcoords="offset points", xytext=(5, 5), ha="center")
        arrow(x, y, vel[i, 1], vel[i, 2], head_width=0.1, head_length=0.2, fc="red", ec="red", length_includes_head=true)
        arrow(x, y, acc[i, 1], acc[i, 2], head_width=0.1, head_length=0.2, fc="green", ec="green", length_includes_head=true)
    end

    box_x = [sim.constants.xi, sim.constants.xf, sim.constants.xf, sim.constants.xi, sim.constants.xi]
    box_y = [sim.constants.yi, sim.constants.yi, sim.constants.yf, sim.constants.yf, sim.constants.yi]
    plot(box_x, box_y, color="black", linestyle="--", label="Box")

    xlabel("x-axis")
    ylabel("y-axis")
    grid(true)
    legend(loc="upper right")
    title("Particle Positions with Velocities and Accelerations")
    show()
end
