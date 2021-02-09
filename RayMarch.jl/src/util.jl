# using GeometryBasics
using Distributions
using LinearAlgebra: dot, cross
using Random

@inline relu(x) = max(x, 0.0)
len(x) = sqrt(sum(x .^ 2))
normalize(x) = x ./ len(x)
direction_and_length(x) = (l = len(x); (x ./ l, l))
rand_uniform(a::Float64, b::Float64, rng::AbstractRNG) = rand(rng, Uniform(a, b))
positive_projection(x, y) = dot(x, y) > 0.0

const xHat  = Vec3(1., 0., 0.)
const yHat  = Vec3(0., 1., 0.)
const zHat  = Vec3(0., 0., 1.)

const Angle = Float64

function rotate_X(p, angle::Angle)
  c = cos(angle)
  s = sin(angle)
  px, py, pz = p
  Vec3(px, c*py - s*pz, s*py + c*pz)
end

function rotate_Y(p, angle::Angle)
  c = cos(angle)
  s = sin(angle)
  px, py, pz = p
  Vec3(c*px + s*pz, py, - s*px+ c*pz)
end

function rotate_Z(p, angle::Angle)
  c = cos(angle)
  s = sin(angle)
  px, py, pz = p
  Vec3(c*px - s*py, s*px+c*py, pz)
end

function sample_cosine_weighted_hemisphere(normal, rng::AbstractRNG)
  u1 = rand(rng)
  u2 = rand(rng)
  uu = normalize(cross(normal, Vec3(0.0, 1.1, 1.1)))
  vv = cross(uu, normal)
  ra = sqrt(u2)
  rx = ra * cos(2.0 * pi * u1)
  ry = ra * sin(2.0 * pi * u1)
  rz = sqrt(1.0 - u2)
  rr = (rx .* uu) + (ry .* vv) + (rz .* normal)
  normalize(rr)
end