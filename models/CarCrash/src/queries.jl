# using OmegaCore
# using Distributions

import Omega
using Omega.Prim: Normal, ~, cond, replace
const randsample = rand
using Statistics: mean

export prob, carjoint

const μspeed = 3.0
const accel = 1 ~ Normal(μspeed, 1)
const car_init_vel = 2 ~ Normal(10.0, 2.0)
const obs_x_pos = 4 ~ Normal(38.0, 0.01)
# const CAR_INIT_VEL =10.0

"Raandom variable over simulations"
function simrv(ω) 
  decel = 9.0
  timestep = 0.1
  x = obs_x_pos(ω)
  # x = 38
  simulate_scene(; ACCEL = accel(ω),
                   DECEL = decel,
                   CAR_INIT_VEL = car_init_vel(ω),
                   timestep = timestep,
                   OBSTRUCTION_POS = VecE2(x, -7.0))
end

crashed(ω) = check_collision(simrv(ω)[1])

animation_(ω) = animate_scene(simrv(ω)...)

animation_move_obs = replace(~ animation_, obs_x_pos => 34)

mindist(ω) = min_distance_btn_car_and_ped(simrv(ω)[1])

prob(x; n = 1000) = mean(rand(x, n; alg = Omega.RejectionSample))

function run_cf()
  # cf_crash = (crashed |ᵈ (accel => 1.0)) |ᶜ crashed
  crashed_ = 3 ~ crashed
  a = 0.01
  cf_crash = cond(replace(crashed_, accel => a), crashed_)
  println("Given that driver crashed, the probability they would have crashed
  had their acceleration been $a ", prob(cf_crash))

  cf_crash = cond(replace(crashed_, obs_x_pos => 34), crashed_)
  println("Given that driver crashed, the probability they would have crashed
  had their the object moved", prob(cf_crash))
end

# carjoint = @joint(accel, car_init_vel, crashed, sim)