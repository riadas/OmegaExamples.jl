using CarCrash
using RayMarch
using RayMarch: xHat, yHat, zHat, Light, PassiveObject, Wall, Matte, Block, Sphere, Mirror, Params, Camera, takePicture
using Images 
using AutomotiveDrivingModels
using Statistics
import ForwardDiff
import ReverseDiff

export render_car_scenes, render_car_scene, render_3D_from_2D

"Render 3D scenes (multiple) with car, pedestrian, and obstruction in given positions"
function render_car_scenes(car_positions::Array{Vec3}, 
                           ped_positions::Array{Vec3};
                           obs_pos::Vec3=Vec3(-0.9, -1.25, -.75),
                           num_samples::Int64=50,
                           max_bounces::Int64=10,
                           camera_dim::Int64=100,
                           camera_pos::Vec3=Vec3(0.0, 0.0, 15.0),
                           camera_halfwidth::Float64=0.3,
                           camera_sensorDist::Float64=1.0,
                           file_name::String="car")
  num_timesteps = length(car_pos)

  for i in 1:num_timesteps
    render_car_scene(car_pos=car_positions[i],
                     ped_pos=ped_positions[i],
                     obs_pos=obs_pos,
                     num_samples=num_samples,
                     max_bounces=max_bounces,
                     camera_dim=camera_dim,
                     camera_pos=camera_pos,
                     camera_halfwidth=camera_halfwidth,
                     camera_sensorDist=camera_sensorDist,
                     file_name=string(file_name, "_", i, ".", "png"))
  end
end

function car_scene(; car_pos, ped_pos, obs_pos)
  # define scene
  lightColor     = Vec3([0.7, 0.7, 0.7])
  leftWallColor  = 1.5 .* Vec3([0.611, 0.0555, 0.062])
  carColor       = 1.5 .* Vec3([0.611, 0.0555, 0.062])
  rightWallColor = 1.5 .* Vec3([0.117, 0.4125, 0.115])
  whiteWallColor = Vec3([255.0, 239.0, 196.0]) / 255.0
  blockColor     = Vec3([200.0, 200.0, 255.0]) / 255.0
  blue           = Vec3([0.0,   191.0, 255.0]) / 255.0
  yellow         = Vec3([255.0, 255.0, 102.0]) / 255.0
  asphault       = Vec3([127.0, 127.0, 127.0]) / 255.0
  black          = Vec3([20.0,   20.0,  20.0]) / 255.0

  car_x, car_y, car_z = car_pos
  ped_x, ped_y, ped_z = ped_pos
  obs_x, obs_y, obs_z = obs_pos

  #=  
  car = [PassiveObject(Block(Vec3([     car_x, car_y, car_z]), Vec3([0.9, 0.7, 0.5]) .* 0.5 .* (1.44/1.7), 0), Matte(carColor)), # car body
        PassiveObject(Block(Vec3([car_x - 0.65 * (1.44/1.7), car_y, car_z]), Vec3([0.4, 0.3, 0.5]) .* 0.5 .* (1.44/1.7), 0), Matte(carColor)), # hood 
        PassiveObject(Block(Vec3([car_x + 0.65 * (1.44/1.7), car_y, car_z]), Vec3([0.4, 0.3, 0.5]) .* 0.5 .* (1.44/1.7), 0), Matte(carColor))] # trunk
        
  # simplified pedestrian: no head or limbs (just body)
  ped = [PassiveObject(Block(Vec3([ped_x,   ped_y - 0.05, ped_z]), Vec3([0.2, 0.3, 0.3]) .* 0.5, 0), Matte(yellow)), # body
         PassiveObject(Block([ped_x - 0.05,  ped_y - 0.6, ped_z], [0.045, 0.3, 0.1] .* 0.5, 0), Matte(yellow)), # left leg
         PassiveObject(Block([ped_x + 0.05,  ped_y - 0.6, ped_z], [0.045, 0.3, 0.1] .* 0.5, 0), Matte(yellow)), # right leg
         PassiveObject(Block([ped_x - 0.125,      ped_y, ped_z], [0.05, 0.2, 0.095] .* 0.5, 0), Matte(yellow)), # left arm
         PassiveObject(Block([ped_x + 0.125,      ped_y, ped_z], [0.05, 0.2, 0.095] .* 0.5, 0), Matte(yellow)), # right arm
         PassiveObject(Sphere([ped_x,       ped_y + 0.4, ped_z], 0.125), Matte(yellow)) ] # head
  
  obstruction = [PassiveObject(Block(Vec3([obs_x, obs_y, obs_z]), Vec3([1.5, 0.75, 0.75]), 0), Matte(blue))]

  theScene = [Light(Vec3([-1.5, 2.0, 1.6]), 1.0, lightColor),
              PassiveObject(Wall(yHat, 2.0), Matte(asphault)),
              obstruction...,
              car...,
              ped...,
              yellow_lines...]
  =#
  
  
  car_z = car_z + 2.0
  ped_z = ped_z + 2.0
  obs_z = obs_z + 2.0

  ped_x = -0.5 
  ped_z = 2.5

  car2 = [PassiveObject(Block(Vec3([-1.0, car_y, 6.5]), Vec3([1.0, 0.3, 0.7]) .* 0.5 .* (1.44/1.7), pi), Matte(carColor))] # hood
  ped2 = [PassiveObject(Block([ped_x,                   ped_y, ped_z], [0.1, 0.3*2/3, 0.1], 0), Matte(yellow)), # body
          PassiveObject(Block([ped_x,  ped_y - 0.6*2/3, ped_z - 0.055], [0.1, 0.3*2/3, 0.045], 0), Matte(yellow)), # left leg
          PassiveObject(Block([ped_x,  ped_y - 0.6*2/3, ped_z + 0.055], [0.1, 0.3*2/3, 0.045], 0), Matte(yellow)), # right leg
          PassiveObject(Block([ped_x,            ped_y, ped_z - 0.125], [0.095, 0.2*2/3, 0.025], 0), Matte(yellow)), # left arm
          PassiveObject(Block([ped_x,            ped_y, ped_z + 0.125], [0.095, 0.2*2/3, 0.025], 0), Matte(yellow)), # right arm
          PassiveObject(Sphere([ped_x, ped_y + 0.4*2/3, ped_z], 0.125*2/3), Matte(yellow))] # head
obstruction2 = [PassiveObject(Block(Vec3([5.0, obs_y, 7]), Vec3([5.5, 0.75, 4.0]),0), Matte(blue))]

yellow_lines = [PassiveObject(Block(Vec3([-1.5, -3, 0]), Vec3([0.05, 1.01, 0.6]), 0), Matte(yellow)),
                PassiveObject(Block(Vec3([-1.5, -3, 2]), Vec3([0.05, 1.01, 0.6]), 0), Matte(yellow)),
                PassiveObject(Block(Vec3([-1.5, -3, 4]), Vec3([0.05, 1.01, 0.6]), 0), Matte(yellow))]

theScene2 = [Light(Vec3([-1.5, 1.25, -0.5]), 1.0, lightColor),
             Light(Vec3([-1.0, 2.5, 7]), 1.0, lightColor),
             Light(Vec3([5.0, 0.5, 1.0]), 1.0, lightColor),
             PassiveObject(Wall(yHat, 2.0), Matte(asphault)),
             car2..., 
             ped2..., 
             obstruction2..., yellow_lines...]
theScene = theScene2

end
# Light(Vec3([-2.5, 0.5, -0.5]), 0.5, lightColor),
# Light(Vec3([-2.5, 0.75, -0.2]), 1.0, lightColor),
# Light(Vec3([1.0, 1.6, 2.5]), 1.0, lightColor),
# Light(Vec3([-2.5, 0.75, 1.0]), 1.0, lightColor),

"""Render 3D scene (single) with car, pedestrian, and obstruction in given positions"""
function render_car_scene(;car_pos = Vec3(3.0, -1.5,  0.6), # x ranges from -3 to 3
                           ped_pos = Vec3(-3.0, -1.4, -0.9), # z ranges from -0.9 +
                           obs_pos = Vec3(-0.9, -1.25, -.75), # stationary
                           num_samples = 50,
                           max_bounces = 10,
                           camera_dim = 100,
                           camera_pos = Vec3(0.0, 0.0, 15.0),
                           camera_halfwidth = 0.3,
                           camera_sensorDist = 1.0,
                           file_name ="car",
                           save_bool = false,
                           progress = false)

  raymarch_params = Params(num_samples, max_bounces, true)
  camera = Camera(camera_dim, camera_pos, camera_halfwidth, camera_sensorDist)

  theScene = car_scene(; car_pos = car_pos, ped_pos = ped_pos, obs_pos = obs_pos)
  @time image_matrix = takePicture(raymarch_params, theScene, camera)

  if save_bool
    save(string(file_name, ".png"), imgview(image_matrix))
  end
  image_matrix
end

function imgview(image_matrix)
  final_image = rescale_image(image_matrix)
  colorview(RGB, final_image)
end

function rescale_image(image_matrix::Array{Float64,3})
  m = maximum(map(x -> x > 1 ? -Inf64 : x, image_matrix/mean(image_matrix)))
  n = minimum(map(x -> x > 1 ? Inf64 : x, image_matrix/mean(image_matrix)))

  a = 255.0/(m - n)
  b = -a*n

  thresholded_image = map(x -> x > 1 ? Inf64 : x, image_matrix/mean(image_matrix)) .* a .+ b
  map(x -> x == Inf64 ? 255.0 : x, thresholded_image)./255
end

function render_3D_from_2D(scene::Scene; obs_pos::Vec3=Vec3([-0.9, -1.25, -.75]),
                                         num_samples::Int64=50,
                                         max_bounces::Int64=10,
                                         camera_dim::Int64=100,
                                         camera_pos::Vec3=Vec3([0.0, 0.0, 15.0]),
                                         camera_halfwidth::Float64=0.3,
                                         camera_sensorDist::Float64=1.0,
                                         file_name::String="car",
                                         save_bool = false)

  car_pos_x = posg(scene[1].state.veh).x
  ped_pos_y = posg(scene[2].state.veh).y

  scale_factor = 6.0/20.0

  car_pos_x_3D = -(car_pos_x - 30.0) * scale_factor + 3.0
  ped_pos_z_3D = ped_pos_y * scale_factor + 0.6

  car_pos_3D = Vec3([car_pos_x_3D, -1.5,  0.6])
  ped_pos_3D = Vec3([-3.0, -1.25, ped_pos_z_3D])

  render_car_scene(car_pos = car_pos_3D, # x ranges from -3 to 3
                   ped_pos = ped_pos_3D, # z ranges from -0.9 +
                   obs_pos = obs_pos, # stationary
                   num_samples = num_samples,
                   max_bounces = max_bounces,
                   camera_dim = camera_dim,
                   camera_pos = camera_pos,
                   camera_halfwidth = camera_halfwidth,
                   camera_sensorDist = camera_sensorDist,
                   file_name = file_name,
                   save_bool = save_bool)
end

function render_3D_from_2D(scenes::AbstractArray; kwargs...)
  num_timesteps = length(scenes)

  for i in 1:num_timesteps
    render_3D_from_2D(scenes[i], kwargs...)
  end
end


function fast_params()
  (car_pos = Vec3(3.0, -1.5,  0.6),
   ped_pos = Vec3(-3.0, -1.4, -0.9),
   obs_pos = Vec3(-0.9, -1.25, -.75),
   num_samples = 5,
   max_bounces = 3,
   camera_dim = 100,
   camera_pos = Vec3([0.0, 0.0, 15.0]),
   camera_halfwidth = 0.3,
   camera_sensorDist = 1.0,
   file_name = "car")
end