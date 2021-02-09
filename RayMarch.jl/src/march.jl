# using GeometryBasics
using Zygote
using Random 

# const Vec3 = Array{Float64}
const Image = Array{Float64, 3}
const Distance = Float64

const Position = Vec3
const Direction = Vec3  ## Should be normalized. TODO: use a newtype wrapper

const Color = Vec3
const BlockHalfWidths = Vec3
const Radius = Float64
const Radiance = Color # Hmm..

abstract type ObjectGeom end

struct Wall <: ObjectGeom
  direction::Direction
  distance::Distance
end

struct Block <: ObjectGeom
  position::Position
  blockhalfwidths::BlockHalfWidths
  angle::Angle
end

struct Sphere <: ObjectGeom
  position::Position
  radius::Radius
end

abstract type Surface end

struct Matte <: Surface
  color::Color
end

struct Mirror <: Surface end

const OrientedSurface = Tuple{Direction, Surface}

abstract type Object end

struct PassiveObject{O <: ObjectGeom, S <: Surface} <: Object
  object_geom::O
  surface::S
end

struct Light <: Object
  position::Position
  hw::Float64
  radiance::Radiance
end

const Ray = Tuple{Position, Direction}
const Filter = Color

struct Params
  num_samples::Int
  max_bounces::Int
  shared_seed::Bool
end

const Scene = Vector{Object}

function sample_reflection(oriented_surface::OrientedSurface, ray::Ray, rng::AbstractRNG)::Ray 
  nor, surf = oriented_surface
  pos, dir = ray
  if surf isa Matte
    new_dir = sample_cosine_weighted_hemisphere(nor, rng)
  else
    new_dir = dir - (2.0 * dot(dir, nor)) * nor
  end
  (pos, new_dir)
end

function prob_reflection(oriented_surface::OrientedSurface, ray1::Ray, ray2::Ray)::Float64 
  nor, surf = oriented_surface
  _, outRayDir = ray2
  if surf isa Matte
    relu(dot(nor, outRayDir))
  else
    0.0
  end
end

function apply_filter(filter::Filter, radiance::Radiance)::Radiance 
  filter .* radiance
end

function surface_filter(filter::Filter, surf::Surface)::Filter 
  if surf isa Matte
    filter .* surf.color
  else
    filter
  end
end

"Signed distance from position `pos` to object"
function sd_object(pos, obj::Object) 
  if obj isa PassiveObject
    object_geom = obj.object_geom
    if object_geom isa Wall
      nor = object_geom.direction
      d = object_geom.distance
      d + dot(nor, pos)
    elseif object_geom isa Block
      blockPos = object_geom.position
      halfWidths = object_geom.blockhalfwidths
      angle = object_geom.angle
      newPos = rotate_Y(pos - blockPos, angle)
      len(map(relu, map(abs, newPos) - halfWidths))
    else # object_geom isa Sphere
      spherePos = object_geom.position
      r = object_geom.radius

      newPos = pos - spherePos
      relu(len(newPos) - r)
    end
  else # obj isa Light
    squarePos = obj.position
    hw = obj.hw

    newPos = pos - squarePos
    # halfWidths = [hw, 0.01, hw]
    halfWidths = Vec3(hw, 0.01, hw)
    len(map(relu, map(abs, newPos) - halfWidths))
  end
end

function t1(object_geom::Wall, pos::Position)
  nor = object_geom.direction
  d = object_geom.distance
  d + dot(nor, pos)
end

function t2(object_geom::Block, pos::Position)
  blockPos = object_geom.position
  halfWidths = object_geom.blockhalfwidths
  angle = object_geom.angle
  newPos = rotate_Y(pos - blockPos, angle)
  len(Vec3(relu(abs(newPos[1]) - halfWidths[1]),
           relu(abs(newPos[2]) - halfWidths[2]),
           relu(abs(newPos[3]) - halfWidths[3])))
  # len(map(relu, map(abs, newPos) - halfWidths))
end

function t3(object_geom::Sphere, pos::Position)
  spherePos = object_geom.position
  r = object_geom.radius
  newPos = pos - spherePos
  relu(len(newPos) - r)
end

function t4(obj::Light, pos::Position)
  squarePos = obj.position
  hw = obj.hw

  newPos = pos - squarePos
  # halfWidths = [hw, 0.01, hw]
  halfWidths = Vec3(hw, 0.01, hw)
  len(map(relu, map(abs, newPos) - halfWidths))

end

function handle_passive(s, pos)
  if s isa Wall
    t1(s, pos)
  elseif s isa Block
    t2(s, pos)
  elseif s isa Sphere
    t3(s, pos)
  end
end

function sd_object(pos, scene, i)
  s = scene[i]
  # o = sd_object(pos, s)
  if s isa PassiveObject
    # @show fieldnames(typeof(s))
    handle_passive(s.object_geom, pos)
  else
    t4(s, pos)
  end
end

function sdScene(scene::Scene, pos::Position)
  # tuples = []
  min_ = Inf
  i_min = 1
  for i in 1:length(scene)
    # s = scene[i]
    d = sd_object(pos, scene, i)
    if d < min_
      i_min = i
      min_ = d
    end
  end
  (i_min, min_)
end

function calcNormal(obj::Object, pos::Position)::Direction
  # pos = [pos[1], pos[2], pos[3]]
  # f(x) = sd_object(x, obj)
  # grads = gradient(x -> sd_object(x, obj), pos) # produces tuple
  # grad = grads[1]
  # normalize(Vec3(grad[1]))
  # grad = ForwardDiff.gradient(f, pos)
  # Vec3(grad[1], grad[2], grad[3])
  Vec3(rand(), rand(), rand())

end

# ----- Start: Define RayMarchResult ----- #
abstract type RayMarchResult end

"Denotes that `ray` hit an object"
struct HitObj <: RayMarchResult
  ray::Ray
  oriented_surface::OrientedSurface
end

struct HitLight <: RayMarchResult
  radiance::Radiance
end

struct HitNothing <: RayMarchResult end

# ----- End: Define RayMarchResult ----- #


"""
`raymarch(scene::Scene, ray::Ray; maxiters = 50)`

Ray marchinges `scene` for a single `ray`
"""
function raymarch(scene::Scene, ray::Ray; maxiters = 50)::RayMarchResult 

  tol = 0.01
  startLength = 10.0 * tol # trying to escape the current surface
  rayOrigin, rayDir = ray

  ray_length = 10.0 * tol
  defaultOutput = HitNothing()
  for i in 1:maxiters
    rayPos = rayOrigin + ray_length .* rayDir
    # obj = scene[1]
    # d = 1.2
    q = sdScene(scene, rayPos) # 29
    obj = scene[q[1]]
    d = q[2]
    
    # 0.9 ensures we come close to the surface but don't touch it
    ray_length = ray_length + 0.9 * d

    if d < tol
      surfNorm = calcNormal(obj, rayPos)
      if !(positive_projection(rayDir, surfNorm))
        if obj isa PassiveObject
          surf = obj.surface
          return HitObj((rayPos, rayDir), (surfNorm, surf)) # 6
        else # obj isa Light
          radiance = obj.radiance
          # println(radiance)
          return HitLight(radiance)
        end
        break
      end
    end
  end
  defaultOutput
end

function rayDirectRadiance(scene::Scene, ray::Ray)::Radiance 
  result = raymarch(scene, ray)
  if result isa HitLight
    # println(radiance)
    result.radiance
  elseif result isa HitNothing
    # [0.0, 0.0, 0.0] # color zero
    Vec3(0.0, 0.0, 0.0)
  else # result isa HitObj
    # [0.0, 0.0, 0.0] # color zero
    Vec3(0.0, 0.0, 0.0)
  end
end

function sampleSquare(hw::Float64, rng::AbstractRNG)::Position
  x = rand_uniform(-hw, hw, rng)
  z = rand_uniform(-hw, hw, rng)
  # [x, 0.0, z]
  Vec3(x, 0.0, z)
end

function sampleLightRadiance(scene::Scene, osurf::OrientedSurface, inRay::Ray, rng::AbstractRNG)::Radiance 
  surfNor, surf = osurf
  rayPos, _ = inRay

  # radiance = [0.0, 0.0, 0.0]
  radiance = Vec3(0.0, 0.0, 0.0)
  for object in scene
    if object isa Light
      lightPos = object.position
      hw = object.hw
      dirToLight, distToLight = direction_and_length(lightPos + sampleSquare(hw, rng) - rayPos)
      if positive_projection(dirToLight, surfNor)
        fracSolidAngle = relu(dot(dirToLight, yHat)) * (hw^2) / (pi * (distToLight)^2)
        outRay = (rayPos, dirToLight)
        coeff = fracSolidAngle * prob_reflection(osurf, inRay, outRay)
        radiance += coeff * rayDirectRadiance(scene, outRay)
      end
    end
  end
  # println(radiance)
  radiance
end

function trace(params::Params, scene::Scene, initRay::Ray, rng::AbstractRNG)::Color 
  # filter = [1.0, 1.0, 1.0]
  filter = Vec3(1.0, 1.0, 1.0)
  ray = initRay
  radiance = Vec3(0.0, 0.0, 0.0)
  # radiance = [0.0, 0.0, 0.0]
  for i in 1:(params.max_bounces)
    result = raymarch(scene, ray)
    if result isa HitNothing           
      break
    elseif result isa HitLight
      if i == 1
        # println("HERE?")
        intensity = result.radiance
        radiance += intensity
      end
      break
    else # result isa HitObj
      incidentRay = result.ray
      osurf = result.oriented_surface
      
      lightRadiance = sampleLightRadiance(scene, osurf, incidentRay, rng)
      ray = sample_reflection(osurf, incidentRay, rng)
      filter = surface_filter(filter, osurf[2])
      radiance += apply_filter(filter, lightRadiance)
    end
  end
  radiance
end

mutable struct Camera 
  numPix::Int
  pos::Position
  halfWidth::Float64
  sensorDist::Float64
end

@inline function sample_average(sample::Function, x::Float64, y::Float64, n::Int, rng::AbstractRNG)
  # tot = [0.0, 0.0, 0.0]
  Vec3
  tot = Vec3(0.0, 0.0, 0.0)
  for i = 1:n
    # @show size(sample(x, y, rng))
    tot = tot .+ sample(x, y, rng)
  end
  tot / n
  # sum([sample(x, y, rng) for i in 1:n])/n
end

function takePicture(params::Params, scene::Scene, camera::Camera; rng = MersenneTwister(0))::Image
  n = camera.numPix
  # start originally in cameraRays
  halfWidth = camera.halfWidth
  pixHalfWidth = halfWidth/n 
  ys = reverse(linspace(n, -halfWidth, halfWidth))
  xs = linspace(n, -halfWidth, halfWidth)  
  image = zeros(Float64, 3, n, n)

  function sampleRayColor(x::Float64, y::Float64, rng::AbstractRNG)::Color
    trace(params, scene, (camera.pos, normalize(Vec3(x, y, -camera.sensorDist))), rng)
  end
  # end originally in camera rays

  for i in 1:n
    println(i)
    for j in 1:n
      color = sample_average(sampleRayColor, xs[j], ys[i], params.num_samples, rng)
      @inbounds image[1, i, j] = color[1]
      @inbounds image[2, i, j] = color[2]
      @inbounds image[3, i, j] = color[3]
    end
  end
  image
end

function linspace(n::Int64, low::Float64, high::Float64)::Array{Float64}
  dx = (high - low)/n
  output = zeros(n)
  for i in 1:n
    output[i] = low + i * dx
  end
  output
end