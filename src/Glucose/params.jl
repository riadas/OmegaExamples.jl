using Omega
using SuParameters
using RunTools
using Random

mutable struct Params
  binsize::Int # [1, 5, 10, 20]
  batchsize::Int # [6, 8, 10, 12, 15]
  maxiters::Int # [150, 300]
  lr::Float64 # [0.01 and smaller]
end

function f(rng)
  rand(rng)
end

function simulate(p)

end

function hyper(; params = Params(), n = 10)
  params_ = merge(p(), params)
  paramsamples = rand(params_, n)
  display.(paramsamples)
  control(infer, paramsamples)
end