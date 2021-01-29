module CarCrash3D

using RayMarch
using RayMarch: xHat, yHat, zHat, Light, PassiveObject, Wall, Matte, Block, Sphere, Mirror, Params, Camera, takePicture
using Images 
using BenchmarkTools

export render_car_scenes, render_car_scene

function render_car_scenes(car_positions::Array{Vec3}, 
                           ped_positions::Array{Vec3};
                           obs_pos::Vec3=[2.0, -0.5,  -4.0],
                           num_samples::Int64=50,
                           max_bounces::Int64=10,
                           camera_dim::Int64=100,
                           camera_pos::Vec3=[0.0, 0.0, 10.0],
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

function render_car_scene(;car_pos::Vec3=[1.0, -1.2,  1.2],
                           ped_pos::Vec3=[-1.5, -1.1, 1.2],
                           obs_pos::Vec3=[2.0, -0.5,  -4.0],
                           num_samples::Int64=50,
                           max_bounces::Int64=10,
                           camera_dim::Int64=100,
                           camera_pos::Vec3=[0.0, 0.0, 10.0],
                           camera_halfwidth::Float64=0.3,
                           camera_sensorDist::Float64=1.0,
                           file_name::String="car")

  raymarch_params = Params(num_samples, max_bounces, true)
  camera = Camera(camera_dim, camera_pos, camera_halfwidth, camera_sensorDist)

  # define scene
  lightColor     = [1.0, 1.0, 1.0]
  leftWallColor  = 1.5 .* [0.611, 0.0555, 0.062]
  carColor       = 1.5 .* [0.611, 0.0555, 0.062]
  rightWallColor = 1.5 .* [0.117, 0.4125, 0.115]
  whiteWallColor = [255.0, 239.0, 196.0] / 255.0
  blockColor     = [200.0, 200.0, 255.0] / 255.0
  blue           = [0.0,   191.0, 255.0] / 255.0
  yellow         = [255.0, 255.0, 102.0] / 255.0

  car_x, car_y, car_z = car_pos
  ped_x, ped_y, ped_z = ped_pos
  obs_x, obs_y, obs_z = obs_pos

  car = [PassiveObject(Block([car_x,      car_y,  car_z], [0.9, 0.7, 0.5], 0), Matte(carColor)), # car body
        PassiveObject(Block([car_x - 1.3, car_y,  car_z], [0.4, 0.3, 0.5], 0), Matte(carColor)), # hood 
        PassiveObject(Block([car_x + 1.3, car_y,  car_z], [0.4, 0.3, 0.5], 0), Matte(carColor))] # trunk

  # simplified pedestrian: no head or limbs (just body)
  ped = [PassiveObject(Block([ped_x, ped_y - 0.1, ped_z], [0.1, 0.4, 0.1], 0), Matte(yellow)),] # body
        # PassiveObject(Block([ped_x - 0.05, ped_y - 0.6, ped_z], [0.045, 0.3, 0.1], 0), Matte(yellow)), # left leg
        # PassiveObject(Block([ped_x + 0.05, ped_y - 0.6, ped_z], [0.045, 0.3, 0.1], 0), Matte(yellow)),] # right leg
        # PassiveObject(Block([ped_x - 0.15,       ped_y, ped_z], [0.045, 0.2, 0.095], 0), Matte(yellow)), # left arm
        # PassiveObject(Block([ped_x + 0.15,       ped_y, ped_z], [0.045, 0.2, 0.095], 0), Matte(yellow)), # right arm
        # PassiveObject(Sphere([ped_x,       ped_y + 0.4, ped_z], 0.125), Matte(yellow))] # head

  #= full pedestrian
  ped = [PassiveObject(Block([ped_x,             ped_y, ped_z], [0.1, 0.3, 0.1], 0), Matte(yellow)), # body
         PassiveObject(Block([ped_x - 0.05, ped_y - 0.6, ped_z], [0.045, 0.3, 0.1], 0), Matte(yellow)), # left leg
         PassiveObject(Block([ped_x + 0.05, ped_y - 0.6, ped_z], [0.045, 0.3, 0.1], 0), Matte(yellow)),] # right leg
         # PassiveObject(Block([ped_x - 0.15,       ped_y, ped_z], [0.045, 0.2, 0.095], 0), Matte(yellow)), # left arm
         # PassiveObject(Block([ped_x + 0.15,       ped_y, ped_z], [0.045, 0.2, 0.095], 0), Matte(yellow)), # right arm
         # PassiveObject(Sphere([ped_x,       ped_y + 0.4, ped_z], 0.125), Matte(yellow))] # head
  =#

  obstruction = [PassiveObject(Block([obs_x, obs_y, obs_z], [3.0, 1.5, 3.0], 0), Matte(blue))]

  theScene = theScene = [Light([-1.0, 1.9, 2.5], 0.75, lightColor),
                         PassiveObject(Wall(yHat, 2.0), Matte(whiteWallColor)),
                         obstruction...,
                         car...,
                         ped...]
  # run raymarch
  @time begin
    image_matrix = takePicture(raymarch_params, theScene, camera)
  end

  # rescale and save
  final_image = rescale_image(image_matrix)
  save(string(file_name, ".png"), colorview(RGB, final_image))
end 

function rescale_image(image_matrix::Array{Float64,3})
  m = maximum(map(x -> x > 1 ? -Inf64 : x, image_matrix/mean(image_matrix)))
  n = minimum(map(x -> x > 1 ? Inf64 : x, image_matrix/mean(image_matrix)))

  a = 255.0/(m - n)
  b = -a*n

  thresholded_image = map(x -> x > 1 ? Inf64 : x, image_matrix/mean(image_matrix)) .* a .+ b
  map(x -> x == Inf64 ? 255.0 : x, thresholded_image)./255
end

function fast_params()
  (car_pos = Vec3([1.0, -1.2,  1.2]),
   ped_pos = Vec3([-1.5, -1.1, 1.2]),
   obs_pos = Vec3([2.0, -0.5,  -4.0]),
   num_samples = 5,
   max_bounces = 3,
   camera_dim = 1000,
   camera_pos = Vec3([0.0, 0.0, 10.0]),
   camera_halfwidth = 0.3,
   camera_sensorDist = 1.0,
   file_name = "car")
end

end