using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC, MCMCChains
using JLD, StatsPlots

include("data.jl")

function run_model(var::String, bin_size::Int64)
  u0, ode_data = prepare_data(var, bin_size)
  datasize = length(ode_data[1,:])
  tspan = (0.0, Float64(datasize) - 1)
  tsteps = range(tspan[1], tspan[2], length = datasize)
  
  println("u0")
  println(size(u0))
  println("ode_data")
  println(size(ode_data))

  # ----- define Neural ODE architecture
  dudt2 = FastChain((x, p) -> x.^3,
                    FastDense(2, 25, swish), # FastDense(2, 50, tanh),
                    FastDense(25, 2)) # FastDense(2, 50, tanh),
  prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

  # ----- define loss function for Neural ODE
  function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
  end

  function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
  end

  # ----- define Hamiltonian log density and gradient 
  l(θ) = -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)

  function dldθ(θ)
    x,lambda = Flux.Zygote.pullback(l,θ)
    grad = first(lambda(1))
    return x, grad
  end

  # ----- define step size adaptor function and sampler
  metric  = DiagEuclideanMetric(length(prob_neuralode.p))

  h = Hamiltonian(metric, l, dldθ)

  integrator = Leapfrog(find_good_stepsize(h, Float64.(prob_neuralode.p)))

  prop = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)

  adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.45, prop.integrator))

  samples, stats = sample(h, prop, Float64.(prob_neuralode.p), 500, adaptor, 500; progress=true)

  losses = map(x-> x[1],[loss_neuralode(samples[i]) for i in 1:length(samples)])

  (samples, losses)  
end

function plot_model_results(samples, losses)
  ##################### PLOTS: LOSSES ###############
  scatter(losses, ylabel = "Loss",  yscale= :log, label = "Architecture1: 500 warmup, 500 sample")

  ################### RETRODICTED PLOTS: TIME SERIES #################
  pl = scatter(tsteps, ode_data[1,:], color = :red, label = "Data: CGM", xlabel = "t", title = "CGM & $(var)")
  scatter!(tsteps, ode_data[2,:], color = :blue, label = "Data: $(var)")

  for k in 1:300
    resol = predict_neuralode(samples[100:end][rand(1:400)])
    plot!(tsteps,resol[1,:], alpha=0.04, color = :red, label = "")
    plot!(tsteps,resol[2,:], alpha=0.04, color = :blue, label = "")
  end

  idx = findmin(losses)[2]
  prediction = predict_neuralode(samples[idx])

  plot!(tsteps,prediction[1,:], color = :black, w = 2, label = "")
  plot!(tsteps,prediction[2,:], color = :black, w = 2, label = "")
  savefig(pl, "glucose_$(var)_bin_$(string(bin_size)).png")
end