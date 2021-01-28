using OmegaCore, Distributions

export prob, carjoint

const μspeed = 3.0
const accel = 1 ~ Normal(μspeed, 1)
const car_init_vel = 2 ~ Normal(10.0, 2.0)
# const CAR_INIT_VEL =10.0

function model(ω) 
  decel = 9.0
  timestep = 0.1
  simulate_scene(; ACCEL = accel(ω),
                   DECEL = decel,
                   CAR_INIT_VEL = car_init_vel(ω),
                   timestep = timestep)
end

crashed(ω) = check_collision(model(ω)[1])

prob(x; n = 1000) = mean(randsample(x, n))

function run_cf()
  cf_crash = (crashed |ᵈ (accel => 1.0)) |ᶜ crashed
  println("Given that driver crashed, the probability they would have crashed
  had their acceleration been 1.0 ", prob(cf_crash))
end

# carjoint = @joint(accel, car_init_vel, crashed, sim)