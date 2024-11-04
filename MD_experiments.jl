include("MD_program.jl")

sim_cons = SimulationConstants()

sim = MDSimulationBase(sim_cons)

println(sim.accelerations)
println(get_potential_energy(sim))
println(get_kinetic_energy(sim))
println(get_force(sim))
plot_points(sim)

sim = MDSimulationBase(sim_cons, "./test_init/reflect_pos.csv", "./test_init/reflect_vel.csv")
step_reflect!(sim; debug=true)


using ProgressMeter

function experiment(; dt=0.01, total_steps=10000, record_interval=100)
    sim_cons = SimulationConstants()
    sim = MDSimulationBase(sim_cons)

    # Initialize vectors to store time series data
    Upot_in_time = Float64[]
    K_in_time = Float64[]
    iT_in_time = Float64[]
    refl_in_time = Int[]
    pos_in_time = Vector{typeof(sim.positions)}()
    vel_in_time = Vector{typeof(sim.velocities)}()
    acc_in_time = Vector{typeof(sim.accelerations)}()

    # Helper function to record the current state
    function record_state()
        push!(Upot_in_time, get_potential_energy(sim, sim.positions))
        push!(K_in_time, get_kinetic_energy_mol(sim))
        push!(iT_in_time, get_instantaneous_temperature(sim))
        push!(refl_in_time, sim.total_reflections)
        push!(pos_in_time, copy(sim.positions))  # copy to prevent reference issues
        push!(vel_in_time, copy(sim.velocities))
        push!(acc_in_time, copy(sim.accelerations))
    end

    # Record initial state
    record_state()

    # Main simulation loop with progress bar
    p = Progress(total_steps; desc="Running experiment: ")
    for _ in 1:total_steps
        step!(sim, dt)
        if sim.total_steps % record_interval == 0
            record_state()
        end
        next!(p)
    end

    # Return results as a NamedTuple (Julia's equivalent to Python dict)
    return (
        potential_energy=Upot_in_time,
        kinetic_energy=K_in_time,
        temperature=iT_in_time,
        reflections=refl_in_time,
        positions=pos_in_time,
        velocities=vel_in_time,
        accelerations=acc_in_time
    )
end

# Run the experiment
result = experiment(dt=0.005, total_steps=10^4, record_interval=20)

using Profile
using PProf

# Clear any existing profile data
Profile.clear()

# Start profiling with a reasonable sample rate
Profile.init(n=10^6, delay=0.001)

# Profile the experiment
@profile experiment(dt=0.005, total_steps=10^2, record_interval=20)

# Export and view profile data with PProf
pprof()  # Opens in default browser

# For saving to file instead of browser view
# pprof(web=false)

# For more detailed CPU profiling
@profile begin
    for _ in 1:5  # Multiple runs to get more data
        experiment(dt=0.005, total_steps=10^4, record_interval=20)
    end
end

# Export with different views
pprof(; webport=8080)  #