using Lens
import OmegaCore
import Omega
using ..CarCrash: simrv
using RayMarch
using ImageView
using JLD2
import FileIO
import ..OmegaExamples
import Omega: ==ₛ, ==ᵣ, SSMH, default_cbs, SSMHLoop

datadir = joinpath(dirname(pathof(OmegaExamples)), "CarCrash3D")

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
function camera_footage(ω; t = 4)
  sim = simrv2(ω)
  # Render the scenes
  i = render_3D_from_2D(sim[1][t]; save = false,
                                   camera_dim = 30,
                                   max_bounces = 5,
                                   num_samples = 10)
  Image(i)                             
end

imshow_footage(ω) = imshow(imgview(camera_footage(ω).x))

function query(; datapath = joinpath(datadir, "image_30_30_4.jld2"),
                 key = "res",
                 n = 10,
                 kwargs...)
  cam = ~ camera_footage
  data = FileIO.load(datapath)[key]
  c = cam ==ₛ Image(data) 
  @leval SSMHLoop => default_cbs(n) Omega.rand(simrv2, c, n; alg = SSMH, kwargs...)
  # SSMHLoop => default_cbs(100)
  # rand(Ω, Climate.Θ_cond_rcd, 100; alg = Replica)
end