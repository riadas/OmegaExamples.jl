using Lens
# import OmegaCore
import Omega
using ..CarCrash: simrv, crashed
using RayMarch
using ImageView
using JLD2
import FileIO
# import ..OmegaExamples
import Omega: ==ₛ, ==ᵣ, SSMH, default_cbs, SSMHLoop

datadir = joinpath(dirname(pathof(CarCrash3D)), "..", "data")
image_30_30_4 = joinpath(datadir, "image_30_30_4.jld2")
image_30_30_20_1 = joinpath(datadir, "image_30_30_20_1.jld2")

export camera_footage
const simrv2 = ~ simrv

struct Image{T}
  x::T
end

function loss(a, b)
  sum(sqrt.((a - b).^2))
end

Omega.d(x::Image, y::Image) = loss(x.x, y.x)

"rv over images generated at times `t`"
function camera_footage(ω; t = 10)
  sim = simrv2(ω)
  # Render the scenes
  i = render_3D_from_2D(sim[1][t]; save = false,
                                   camera_dim = 30,
                                   max_bounces = 5,
                                   num_samples = 10)
  Image(i)                             
end

const wascrash = ~crashed

imshow_footage(ω) = imshow(imgview(camera_footage(ω).x))

function load_data(; key = "res", datapath = image_30_30_20_1)
  data = FileIO.load(datapath)[key]
end

function query(; datapath = image_30_30_20_1,
                 key = "res",
                 n = 100,
                 kwargs...)
  cam = ~ camera_footage
  data = FileIO.load(datapath)[key]
  c = cam ==ₛ Image(data) 
  @leval SSMHLoop => default_cbs(n) Omega.rand(simrv2, c, n; alg = SSMH, kwargs...)
end

function querycf(; datapath = image_30_30_20_1,
                 key = "res",
                 n = 100,
                 kwargs...)
  cam = ~ camera_footage
  data = FileIO.load(datapath)[key]
  c1 = cam ==ₛ Image(data)
  i1 = replace(~ CarCrash.mindist, CarCrash.car_init_vel => 10.0) 
  # i2 = replace(~ CarCrash.mindist, CarCrash.obs_x_pos => 34)
  i2 = CarCrash.car_init_vel
  i3 = ~ CarCrash.mindist
  joint = Omega.randtuple((i1, i2, i3))
  # @assert false
  
  c2 = i3 ==ₛ 0.0
  c = c1 & c2
  # @assert false
  # cf = cond(joint, c)
  # a = @leval SSMHLoop => default_cbs(n) Omega.rand(i1, c, n; alg = SSMH, kwargs...)
  # b = @leval SSMHLoop => default_cbs(n) Omega.rand(i2, c, n; alg = SSMH, kwargs...)
  # (accel = a, obs = b)
  # @assert false

  @leval SSMHLoop => default_cbs(n) Omega.rand(joint, c, n; alg = SSMH, kwargs...)
end