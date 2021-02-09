using RayMarch
using RayMarch: xHat, yHat, zHat, Light, PassiveObject, Wall, Matte, Block, Sphere, Mirror, Params, Camera, takePicture
using Images 
using BenchmarkTools
using RayMarch: Vec3
using ImageView


function test(; test_mode = true
                showimg = false)
  smallDim = 40
  # define colors
  lightColor     = Vec3(1.0, 1.0, 1.0)
  leftWallColor  = 1.5 .* Vec3(0.611, 0.0555, 0.062)
  carColor       = 1.5 .* Vec3(0.611, 0.0555, 0.062)
  rightWallColor = 1.5 .* [0.117, 0.4125, 0.115]
  whiteWallColor = Vec3(255.0, 239.0, 196.0) / 255.0
  blockColor     = Vec3(200.0, 200.0, 255.0) / 255.0
  blue           = Vec3(0.0,   191.0, 255.0) / 255.0

  # define scene: car
  theScene = [Light(Vec3(-1.0, 1.9, 2.5), 0.5, lightColor),
              PassiveObject(Wall(yHat, 2.0), Matte(whiteWallColor)),
              PassiveObject(Wall(-yHat, 2.0), Matte(whiteWallColor)),
              PassiveObject(Block(Vec3(1.5, -0.5,  -4.0), Vec3(3.0, 1.5, 3.0), 0), Matte(blue)),
              PassiveObject(Block(Vec3(1.0, -1.2,  1.2), Vec3(0.9, 0.7, 0.5), 0), Matte(carColor)),
              PassiveObject(Block(Vec3(-0.3, -1.2,  1.2), Vec3(0.4, 0.3, 0.5), 0), Matte(carColor)),
              PassiveObject(Block(Vec3(2.3, -1.2,  1.2), Vec3(0.4, 0.3, 0.5), 0), Matte(carColor))]

  # define params
  defaultparams = Params(50, 10, true)

  # define camera
  default_camera = Camera(250, 10.0 .* zHat, 0.3, 1.0)

  # We change to a small num pix here to reduce the compute needed for tests
  params = defaultparams

  if test_mode
    camera = Camera(smallDim, [0.0, 0.0, 10.0], 0.3, 1.0)
  else 
    camera = default_camera
  end

  image_matrix = takePicture(params, theScene, camera)

  m = maximum(map(x -> x > 1 ? -Inf64 : x, image_matrix/mean(image_matrix)))
  n = minimum(map(x -> x > 1 ? Inf64 : x, image_matrix/mean(image_matrix)))

  a = 255.0/(m - n)
  b = -a*n

  thresholded_image = map(x -> x > 1 ? Inf64 : x, image_matrix/mean(image_matrix)) .* a .+ b
  final_image = map(x -> x == Inf64 ? 255.0 : x, thresholded_image)
  # final_image = image_matrix .* a .+ b
  if showimg
    ImageView.imshow(colorview(RGB, final_image./255))
  end
  final_image
end