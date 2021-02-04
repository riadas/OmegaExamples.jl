using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC, MCMCChains
using JLD, StatsPlots
using BSON
using Random
using Glucose: prepare_all_data_meals_hypo
using NNlib
Core.eval(Main, :(import NNlib))

d = BSON.load("/Users/riadas/Documents/urop/OmegaExamples.jl/1_model13_good.bson")

prob = d[:nn_model]
theta = d[:theta]
best_pred = d[:best_pred]
tspan = (0.0, 1.0)
tsteps = range(tspan[1], tspan[2], length=size(ode_data)[2])
datasize = size(ode_data)[2]
u0, ode_data = prepare_all_data_meals_hypo(10)
exo_data = ode_data[3:4, :]
u0 = u0[1:2]

# old model
function predict(θ)
  _prob = remake(prob, p=θ)
  Array(solve(_prob, Tsit5(), saveat=tsteps, abstol = 1e-9, reltol = 1e-9))
end
# end old model

# intervention
f_intervention = FastChain(FastDense(4, 64, swish),
                    FastDense(64, 32, swish),
                    FastDense(32, 2))
function f_exo(u, p, t)
  scaled_t = Int(round(t*(datasize - 1)/(tspan[2] - tspan[1]) + (1 - (datasize - 1)*tspan[1]/(tspan[2] - tspan[1]))))
  #println("Here")
  #println(scaled_t)
  if scaled_t in [14, 15, 16]
    # @show size(vcat(u..., 4.0, exo_data[1, scaled_t]))
    f_intervention(vcat(u..., 4.0, exo_data[2, scaled_t]), p) # intervening on bolus!
  else
    # @show vcat(u, exo_data[:, scaled_t]...)
    #@show u
    f_intervention(vcat(u, exo_data[:, scaled_t]...), p)
  end
end
prob_intervention = ODEProblem(f_exo, u0, tspan, nothing)

function predict_intervention(θ)
  _prob = remake(prob_intervention, p=θ)
  Array(solve(_prob, Tsit5(), saveat=tsteps, abstol = 1e-9, reltol = 1e-9))
end
# end intervention


# plot
pl1 = plot(tsteps, ode_data[1,:], color = :red, w=1.5, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals")
plot!(pl1, tsteps, ode_data[2,:], color = :blue, w=1.5, label = "Data: Steps")
plot!(pl1, tsteps, ode_data[3,:], color = :green, w=1.5, label = "Data: Bolus")
plot!(pl1, tsteps, ode_data[4,:], color = :purple, w=1.5, label = "Data: Meals")

pl2 = plot(tsteps, ode_data[1,:], color = :red, w=1.5, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals")
plot!(pl2, tsteps, ode_data[2,:], color = :blue, w=1.5, label = "Data: Steps")
plot!(pl2, tsteps, vcat(ode_data[3,1:13]..., 4.0, 4.0, 4.0, ode_data[3, 17:end]...), color = :green, w=1.5, label = "Data: Bolus")
plot!(pl2, tsteps, ode_data[4,:], color = :purple, w=1.5, label = "Data: Meals")

thetas = []
losses1 = []
losses2 = []
#=
        pred = predict(θ)
        # println("SHOW HERE 3")
        # @show size(pred)
        Flux.mse(pred[:, 1:size(y)[2]], y)
=#

dist = Normal()
for i in 1:300
  println(i)
  noise = 0.05*rand(dist, length(theta))
  noisy_theta = theta .+ noise

  push!(thetas, noisy_theta)

  resol1 = predict(noisy_theta)
  resol2 = predict_intervention(noisy_theta)

  push!(losses1, Flux.mse(resol1, ode_data[1:2, :]))
  push!(losses2, Flux.mse(resol2, ode_data[1:2, :]))

  plot!(pl1, tsteps,resol1[1,:], alpha=0.4, color = :red, label = "")
  plot!(pl1, tsteps,resol1[2,:], alpha=0.4, color = :blue, label = "")

  plot!(pl2, tsteps,resol2[1,:], alpha=0.4, color = :red, label = "")
  plot!(pl2, tsteps,resol2[2,:], alpha=0.4, color = :blue, label = "")
end

idx = findmin(losses1)[2]
best_pred_new = predict_intervention(thetas[idx])
plot!(pl1, tsteps, best_pred_new[1, :], w=2, color = :black, label="")
plot!(pl1, tsteps, best_pred_new[2, :], w=2, color = :black, label="")

idx = findmin(losses2)[2]
best_pred_intervene = predict_intervention(thetas[idx])
plot!(pl2, tsteps, best_pred_intervene[1, :], w=2, color = :black, label="")
plot!(pl2, tsteps, best_pred_intervene[2, :], w=2, color = :black, label="")

savefig(pl1, "noisy2.png")
savefig(pl2, "noisy_intervention2.png")
