using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC, MCMCChains
using JLD, StatsPlots
using BSON
using Random
using Glucose: prepare_all_data_meals_hypo
using NNlib

# prepare data
_, ode_data = prepare_all_data_meals_hypo(10)
exo_data = ode_data[3:4, :]
tspan = (0.0, 1.0)
tsteps = range(tspan[1], tspan[2], length=size(ode_data)[2])
datasize = size(ode_data)[2]
u0 = ode_data[1:2, 1] 

glucose_mean, glucose_std = (159.44942528735626, 54.68441830006641)
steps_mean, steps_std = (2.1413793103448278, 2.478409230791181)
bolus_mean, bolus_std = (0.030344827586206897, 0.12658940258683044)
meal_mean, meal_std = (0.8620689655172413, 1.8987615875887272)

# extract optimal weights
d = BSON.load("1_model13_good.bson")
theta = d[:theta]

function no_intervention(f, u, p, t)
  scaled_t = Int(round(t*(datasize - 1)/(tspan[2] - tspan[1]) + (1 - (datasize - 1)*tspan[1]/(tspan[2] - tspan[1]))))
  f(vcat(u, exo_data[:, scaled_t]...), p)
end

function generate_series(weights::AbstractArray=theta, intervene::Function=no_intervention)
  Core.eval(Main, :(import NNlib))

  # un-intervened model
  f = FastChain(FastDense(4, 64, swish),
      FastDense(64, 32, swish),
      FastDense(32, 2))

  # intervened model
  function f_intervene(u, p, t)
    intervene(f, u, p, t)
  end

  prob_intervene = ODEProblem(f_intervene, u0, tspan, nothing)
  
  function predict_intervene(θ)
    _prob = remake(prob_intervene, p=θ)
    Array(solve(_prob, Tsit5(), saveat=tsteps, abstol = 1e-9, reltol = 1e-9))
  end

  Array(vcat(predict_intervene(weights), exo_data))
end

function unnormalize_data(data)
  Array(transpose(hcat(
    data[1,:] .* glucose_std .+ glucose_mean,
    data[2, :] .* steps_std .+ steps_mean,
    data[3, :] .* bolus_std .+ bolus_mean,
    data[4, :] .* meal_std .+ meal_mean
  )))
end