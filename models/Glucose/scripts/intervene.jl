using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC, MCMCChains
using JLD, StatsPlots
using BSON
using Random
using Glucose: prepare_all_data_meals_hypo
using NNlib
using Glucose
using LinearAlgebra
using NNlib
using Distributions

# prepare data
_, ode_data = prepare_all_data_meals_hypo(10)

const glucose_mean, glucose_std = (159.44942528735626, 54.68441830006641)
const steps_mean, steps_std = (2.1413793103448278, 2.478409230791181)
const bolus_mean, bolus_std = (0.030344827586206897, 0.12658940258683044)
const meal_mean, meal_std = (0.8620689655172413, 1.8987615875887272)

# extract optimal weights
d = BSON.load("1_model13_good.bson")
theta = d[:theta]

"Make time t ∈ [l, r] to index 1:n"
function scaleidx(t, lb, ub, n)
  δ = ub - lb
  Int(round(t*(n - 1)/δ + (1 - (n - 1)*lb/δ)))
end

function no_intervention(f, u, p, t)
  scaled_t = scaleidx(t, tspan[1], tspan[2], datasize)
  f(vcat(u, exo_data[:, scaled_t]...), p)
end

"Generate the network"
function gennet()
  f = FastChain(FastDense(4, 64, swish),
                FastDense(64, 32, swish),
                FastDense(32, 2))
end

function predict_intervene(prob_intervene, θ)
  _prob = remake(prob_intervene, p=θ)
  res = solve(_prob, Tsit5(), saveat=tsteps, abstol = 1e-9, reltol = 1e-9)
  Array(res)
end

function generate_series(f, weights::AbstractArray=theta, intervene::Function=no_intervention; exo_data=exo_data)
  # intervened model
  function f_intervene(u, p, t)
    intervene(f, u, p, t)
  end

  prob_intervene = ODEProblem(f_intervene, u0, tspan, nothing)

  Array(vcat(predict_intervene(prob_intervene, weights), exo_data))
end

function unnormalize_data(data)
  Array(transpose(hcat(
    data[1,:] .* glucose_std .+ glucose_mean,
    data[2, :] .* steps_std .+ steps_mean,
    data[3, :] .* bolus_std .+ bolus_mean,
    data[4, :] .* meal_std .+ meal_mean
  )))
end

"""Example"""
function examplep()
  network = gennet()
  exo_data = ode_data[3:4, :]
  tspan = (0.0, 1.0)
  tsteps = range(tspan[1], tspan[2], length=size(ode_data)[2])
  datasize = size(ode_data)[2]
  u0 = ode_data[1:2, 1] 

  new_exo_data = copy(exo_data)
  new_exo_data[2, 8] = 4.0
  new_exo_data[2, 9] = 4.0
  new_exo_data[2, 10] = 4.0
  function f_int(f, u, p, t)
    scaled_t = Int(round(t*(datasize - 1)/(tspan[2] - tspan[1]) + (1 - (datasize - 1)*tspan[1]/(tspan[2] - tspan[1]))))
    if scaled_t in [8, 9, 10]
        f(vcat(u..., exo_data[1, scaled_t], 4.0), p) # intervening on bolus!
    else
        f(vcat(u, exo_data[:, scaled_t]...), p)
    end
  end

  # pl1 = plot(tsteps, ode_data[1,:], color = :red, w=1.5, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals")
  # plot!(pl1, tsteps, ode_data[2,:], color = :blue, w=1.5, label = "Data: Steps")
  # plot!(pl1, tsteps, ode_data[3,:], color = :green, w=1.5, label = "Data: Bolus")
  # plot!(pl1, tsteps, ode_data[4,:], color = :purple, w=1.5, label = "Data: Meals")

  # pl2 = plot(tsteps, ode_data[1,:], color = :red, w=1.5, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals")
  # plot!(pl2, tsteps, ode_data[2,:], color = :blue, w=1.5, label = "Data: Steps")
  # plot!(pl2, tsteps, ode_data[3,:], color = :green, w=1.5, label = "Data: Bolus")
  # plot!(pl2, tsteps, vcat(ode_data[4,1:7]..., 4.0, 4.0, 4.0, ode_data[4, 11:end]...), color = :purple, w=1.5, label = "Data: Meals")

  thetas = []
  losses1 = []
  losses2 = []

  resol1s = []
  resol2s = []
  nsamples = 100
  dist = Normal(0, 0.05)
  for i = 1:nsamples
    noise = rand(dist, length(theta))
    noisy_theta = theta .+ noise
    resol1 = generate_series(network, noisy_theta)
    resol2 = generate_series(network, noisy_theta, f_int, exo_data=new_exo_data)
    push!(resol1s, resol1)
    push!(resol2s, resol2)
  end
  resol1s, resol2s, new_exo_data
end

function plotexample()
  resol1s, resol2s, new_exo_data = examplep()
    nsamples = length(resol1s)
  genplot(resol1s, resol2s, tsteps, ode_data, nsamples, exo_data, new_exo_data)
end

function genplot(resol1s, resol2s, tsteps, ode_data, nsamples, exo_data, new_exo_data)
  pl1 = plot(tsteps, ode_data[1,:], seriestype = :scatter, color = :red, w=1.5, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals")
  plot!(pl1, tsteps, ode_data[2,:], seriestype = :scatter, color = :blue, w=1.5, label = "Data: Steps")
  plot!(pl1, tsteps, ode_data[3,:], seriestype = :scatter, color = :green, w=1.5, label = "Data: Bolus")
  plot!(pl1, tsteps, ode_data[4,:], seriestype = :scatter, color = :purple, w=1.5, label = "Data: Meals")

  pl2 = plot(tsteps, ode_data[1,:], seriestype = :scatter, color = :red, w=1.5, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals")
  plot!(pl2, tsteps, ode_data[2,:], seriestype = :scatter, color = :blue, w=1.5, label = "Data: Steps")
  plot!(pl2, tsteps, ode_data[3,:], seriestype = :scatter, color = :green, w=1.5, label = "Data: Bolus")
  plot!(pl2, tsteps, vcat(ode_data[4,1:7]..., 4.0, 4.0, 4.0, ode_data[4, 11:end]...), color = :purple, w=1.5, label = "Data: Meals")

  for i = 1:nsamples
    resol1 = resol1s[i]
    resol2 = resol2s[i]
    plot!(pl1, tsteps,resol1[1,:], alpha=0.1, color = :red, label = "")
    plot!(pl1, tsteps,resol1[2,:], alpha=0.1, color = :blue, label = "")

    plot!(pl2, tsteps,resol2[1,:], alpha=0.1, color = :red, label = "")
    plot!(pl2, tsteps,resol2[2,:], alpha=0.1, color = :blue, label = "")
  end
  pl1, pl2  
end



#   dist = Normal()
#   for i in 1:300
#     println(i)
#     noise = 0.05*rand(dist, length(theta))
#     noisy_theta = theta .+ noise

#     push!(thetas, noisy_theta)

#     resol1 = generate_series(network, noisy_theta)
#     resol2 = generate_series(network, noisy_theta, f_int, exo_data=new_exo_data)

#     @show size(resol1)
#     @show size(resol2)

#     push!(losses1, Flux.mse(resol1[1:2], ode_data[1:2, :]))
#     push!(losses2, Flux.mse(resol2[1:2], ode_data[1:2, :]))

#     plot!(pl1, tsteps,resol1[1,:], alpha=0.4, color = :red, label = "")
#     plot!(pl1, tsteps,resol1[2,:], alpha=0.4, color = :blue, label = "")

#     plot!(pl2, tsteps,resol2[1,:], alpha=0.4, color = :red, label = "")
#     plot!(pl2, tsteps,resol2[2,:], alpha=0.4, color = :blue, label = "")
#   end

  # idx = findmin(losses1)[2]
  # best_pred = predict_intervene(thetas[idx])
  # plot!(pl1, tsteps, best_pred[1, :], w=2, color = :black, label="")
  # plot!(pl1, tsteps, best_pred[2, :], w=2, color = :black, label="")

  # idx = findmin(losses2)[2]
  # best_pred_intervene = predict_intervene(thetas[idx])
  # plot!(pl2, tsteps, best_pred_intervene[1, :], w=2, color = :black, label="")
  # plot!(pl2, tsteps, best_pred_intervene[2, :], w=2, color = :black, label="")

#   savefig(pl1, "noisy_3.png")
#   savefig(pl2, "noisy_intervention_3.png")
#   pl1, pl2  
# end