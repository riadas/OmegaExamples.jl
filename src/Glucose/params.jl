using RunToolsExample: train
using RunToolsExample: allparams
using SuParameters: Params
using RunTools
using Omega: normal, uniform, ciid
include("test.jl")

mutable struct Params
  logdir::String
  binsize::Int # [1, 5, 10, 20]
  batchsize::Int # [6, 8, 10, 12, 15]
  maxiters::Int # [150, 300]
  lr::Float64 # [0.01 and smaller]
end

function runparams()
  φ = Params()
  φ.simname = "train"
  φ.runname = ciid(randrunname)
  φ.tags = ["odeoptim", "first"]
  φ.logdir = ciid(ω -> logdir(runname = φ.runname(ω), tags = φ.tags))
  φ.runfile = joinpath(dirname(@__FILE__), "..", "scripts", "runscript.jl") # TODO must be changed
  φ.gitinfo = current_commit(@__FILE__)
  φ
end

"Optimization Parameters"
function optparams()
  Params((binsize = uniform([1, 5, 10, 15]),
          batchsize = uniform([6, 8, 10, 12, 15]),
          maxiters = uniform([150:100:1050...]),
          lr = uniform([0.001, 0.005, 0.01, 0.05, 0.1]),
          hypoid = uniform([1,2,3])))
end

function train(params::Params)
  samples, losses, best_pred = optim_all_vars_exo(params.binsize, 
                                                  params.batchsize, 
                                                  params.maxiters, 
                                                  params.lr, 
                                                  hypo_id=params.hypoid,
                                                  log_dir=params.logdir) # from test.jl
end

# function netparams()
#   Params(nhidden = uniform(32:64),
#          activation = uniform([relu, elu, selu]))
# end

function allparams()
  # Setup tensorboard
  φ = Params()
  merge(φ, runparams(), optparams()) # netparams()
end

"Run with julia -L hyper.jl -E 'hyper(;)' -- --queue"
function hyper(; params = Params(), n = 1000)
  params_ = optparams()
  paramsamples = rand(params_, n)
  display.(paramsamples)
  control(train, paramsamples)
end