module RayMarch
using StaticArrays
using Zygote
const Vec3 = SVector{3, Float64}
Zygote.@adjoint Vec3(a, b, c) = Vec3(a, b, c), p̄ -> (p̄[1], p̄[2], p̄[3])
import ForwardDiff

include("util.jl")
include("march.jl")
export Vec3

end # module