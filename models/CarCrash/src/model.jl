
using AutomotiveDrivingModels
using AutoViz
using Random
using Interact
using Reel
using Blink

export simulate_scene, animate_scene, check_collision

AutoViz.set_render_mode(:fancy)
AutoViz.colortheme["background"] = colorant"black"

# ----- START: define custom obstruction renderable -----

struct RenderableObstruction
  pos::VecE2
  dim::VecE2
  color::Colorant
end

function AutoViz.add_renderable!(rendermodel::RenderModel, rect::RenderableObstruction)
  # add the desired render instructions to the rendermodel
  add_instruction!(
      rendermodel, AutoViz.render_rect,
      (rect.pos.x, rect.pos.y, rect.dim.x, rect.dim.y, rect.color, true),
      coordinate_system=:scene
  )
  return rendermodel
end

# ----- END: define custom obstruction renderable -----

"""Simulate driving scenario with obstruction"""
function simulate_scene(;NUM_LANES::Int64=3, 
                        CAR_DIMS::VecE2{Float64}=VecE2(4.8, 1.8),
                        CAR_INIT_POS::VecSE2{Float64}=VecSE2(30.0, 0.0, 0.0),
                        CAR_INIT_VEL::Float64=10.0,
                        PED_INIT_POS::VecSE2{Float64}=VecSE2(50.0, -5.0, π/2),
                        PED_INIT_VEL::Float64=3.0,
                        OBSTRUCTION_POS::VecE2{Float64}=VecE2(38.0, -7.0),
                        OBSTRUCTION_DIMS::VecE2{Float64}=VecE2(10.0, 5.0),
                        ACCEL::Float64=3.0,
                        DECEL::Float64=9.0,
                        timestep::Float64=0.1,
                        nticks::Int64=50)

  # define scene
  scene = Scene(Entity{CustomVehicleState,VehicleDef,Int64}, 2)  # allocate a scene for 2 agents
  
  # define roadway
  roadway = gen_straight_roadway(NUM_LANES, 100.0)

  # add car to scene 
  # (speed 10 => no crash; speed 20 => crash)
  push!(scene,
      Entity(CustomVehicleState(VehicleState(CAR_INIT_POS, roadway, CAR_INIT_VEL), 
                                             0,
                                             PED_INIT_POS,
                                             PED_INIT_VEL,
                                             OBSTRUCTION_POS,
                                             OBSTRUCTION_DIMS,
                                             ACCEL,
                                             DECEL), 
      VehicleDef(AgentClass.CAR, CAR_DIMS.x, CAR_DIMS.y), 
      1))

  # add pedestrian to scene
  push!(scene, Entity(
      CustomVehicleState(VehicleState(PED_INIT_POS, roadway, PED_INIT_VEL), 
                                             0,
                                             PED_INIT_POS,
                                             PED_INIT_VEL,
                                             OBSTRUCTION_POS,
                                             OBSTRUCTION_DIMS,
                                             ACCEL,
                                             DECEL),
      VehicleDef(AgentClass.PEDESTRIAN, 1., 1.),
      42))

  # define action models (car and pedestrian)
  models = Dict{Int, Union{IntelligentDriverModel, Tim2DDriver, ConstantPedestrian, CautiousCar}}()
  models[1] = CautiousCar(CarAccelLatLong(0.0, 0.0)) 
  models[42] = ConstantPedestrian(PedestrianAccelLatLong(0.0,0.0)) 

  # array of scene objects (length = nticks)
  scenes = simulate(scene, roadway, models, nticks, timestep)
  obstruction = RenderableObstruction(OBSTRUCTION_POS, OBSTRUCTION_DIMS, colorant"blue")

  # scenes representation with only each agent's pos/vel: 
  # (map(scene -> (scene[1].state.veh.posG, scene[2].state.veh.posG), scenes), obstruction, roadway)

  (scenes, obstruction, roadway) # TODO: return more useful representation of this
end

"""Animate scenes (use: animate_scene(simulate_scene()...))"""
function animate_scene(scenes::AbstractArray, obstruction::RenderableObstruction, roadway::Roadway)
  # create text overlay
  textOverlays = [[TextOverlay(
      text=["Vehicle speed: $(get_by_id(scenes[i], 1).state.veh.v)"],
      font_size=13, pos=VecE2(20.0, 50.0), color=colorant"white",
  )] for i in 1:length(scenes)]

  # create neighbor overlay
  neighborOverlays = [[
      NeighborsOverlay(
          scene=scene, roadway=roadway, target_id=1,
          textparams=TextParams(color=colorant"white")
      )
  ] for (i, scene) in enumerate(scenes)]

  # animate
  w = Window()
  viz = @manipulate for step in 1 : length(scenes)
      render([roadway, obstruction, scenes[step], neighborOverlays[step][1], textOverlays[step][1]])
  end
  body!(w, viz)
end

"""Check if collision occurred during scene"""
function check_collision(scene::Scene)::Bool
  collision_checker(scene[1], scene[2]) # each scene has two agents: car and pedestrian
end

"""Check if collision occurred during sequence of scenes"""
check_collision(scenes::AbstractArray)::Bool = foldl(|, map(check_collision, scenes))

# ----- START: define custom action type and driver model for PEDESTRIAN -----
struct PedestrianAccelLatLong
  a_lat::Float64
  a_lon::Float64
end

function AutomotiveDrivingModels.propagate(veh::Entity{CustomVehicleState,VehicleDef,Int64}, action::PedestrianAccelLatLong, roadway::Roadway, Δt::Float64)
  vehdef = VehicleDef(class(veh.def), AutomotiveDrivingModels.length(veh.def), AutomotiveDrivingModels.width(veh.def))
  
  # state values
  time = veh.state.time
  ped_init_pos = veh.state.ped_init_pos
  ped_vel = veh.state.ped_vel
  obstruction_pos = veh.state.obstruction_pos
  obstruction_dims = veh.state.obstruction_dims
  accel = veh.state.accel
  decel = veh.state.decel
  
  vehicleState = propagate(Entity(veh.state.veh, vehdef, veh.id), LatLonAccel(action.a_lat, action.a_lon), roadway, Δt)
  return CustomVehicleState(vehicleState, veh.state.time + 1, ped_init_pos, ped_vel, obstruction_pos, obstruction_dims, accel, decel)
end

struct ConstantPedestrian <: DriverModel{PedestrianAccelLatLong}
  a::PedestrianAccelLatLong
end

AutomotiveDrivingModels.observe!(model::ConstantPedestrian, scene::Scene{Entity{CustomVehicleState,VehicleDef,Int64}}, roadway::Roadway, egoid::Int64) = model
Base.rand(::AbstractRNG, model::ConstantPedestrian) = model.a

# ----- END: define custom action type and driver model for PEDESTRIAN -----

# ----- START: define custom action type and driver model for CAR -----
struct CarAccelLatLong
  a_lat::Float64
  a_lon::Float64
end

function AutomotiveDrivingModels.propagate(veh::Entity{CustomVehicleState,VehicleDef,Int64}, action::CarAccelLatLong, roadway::Roadway, Δt::Float64)
  vehdef = VehicleDef(class(veh.def), AutomotiveDrivingModels.length(veh.def), AutomotiveDrivingModels.width(veh.def))
  
  # state values
  time = veh.state.time
  ped_init_pos = veh.state.ped_init_pos
  ped_vel = veh.state.ped_vel
  obstruction_pos = veh.state.obstruction_pos
  obstruction_dims = veh.state.obstruction_dims
  accel = veh.state.accel
  decel = veh.state.decel

  carFrontPos = get_front(veh)
  obstructionTopRightX = obstruction_pos.x + obstruction_dims.x
  obstructionTopRightY = obstruction_pos.y + obstruction_dims.y
  pedY = ped_init_pos.y + ped_vel*time * Δt
  
  lineOfSightIntersectPedPathY = carFrontPos.x >= obstructionTopRightX ? 
                                 0 : 
                                 carFrontPos.y - ((ped_init_pos.x - carFrontPos.x)/(obstructionTopRightX- carFrontPos.x))*(carFrontPos.y - obstructionTopRightY)
  if (carFrontPos.x >= obstructionTopRightX || (lineOfSightIntersectPedPathY <= pedY)) && (get_rear(veh).x <= ped_init_pos.x) && (pedY < -obstructionTopRightY)
    # brake!
    vehicleState = propagate(Entity(veh.state.veh, vehdef, veh.id), LatLonAccel(action.a_lat, -decel), roadway, Δt)
    
    # if new velocity is negative, set to 0.0
    if vehicleState.posG < veh.state.veh.posG
      vehicleState = VehicleState(veh.state.veh.posG, roadway, 0.0)
    end 
  else # pedestrian is not visible/not in vehicle's lane
    # TODO: approach desired speed 
    # println(veh.state.veh.v) 
    vehicleState = propagate(Entity(veh.state.veh, vehdef, veh.id), LatLonAccel(0.0, accel), roadway, Δt)
  end
  return CustomVehicleState(vehicleState, veh.state.time + 1, ped_init_pos, ped_vel, obstruction_pos, obstruction_dims, accel, decel)
end

struct CautiousCar <: DriverModel{CarAccelLatLong}
  a::CarAccelLatLong
end

AutomotiveDrivingModels.observe!(model::CautiousCar, scene::Scene{Entity{CustomVehicleState,VehicleDef,Int64}}, roadway::Roadway, egoid::Int64) = model
Base.rand(::AbstractRNG, model::CautiousCar) = model.a

# ----- END: define custom action type and driver model for CAR -----