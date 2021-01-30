using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC, MCMCChains
using JLD, StatsPlots
using LightXML
using Dates

patient_ids = [558, 563, 570, 575, 588, 591]

function parse_data(;file_location::String="../../../OhioT1DM-training/559-ws-training.xml", 
                    data_name::String="glucose_level")

  # xdoc is an instance of XMLDocument, which maintains a tree structure
  xdoc = parse_file(file_location)

  # get the root element
  xroot = root(xdoc)  # an instance of XMLElement
  data_element = find_element(xroot, data_name)

  parsed_data = [[],[]]

  for event in child_nodes(data_element)
    if is_elementnode(event)
       # time in minutes
      push!(parsed_data[1], DateTime(attribute(XMLElement(event), "ts"), "dd-mm-yyyy HH:MM:SS").instant.periods.value/(1000.0 * 60))
      # value
      push!(parsed_data[2], parse(Float64, attribute(XMLElement(event), "value")))
    end
  end

  free(xdoc)

  println(string("length(parsed_data): ", length(parsed_data[1])))
  
  # shift times to begin at t=0
  start_time = minimum(parsed_data[1])
  parsed_data = (map(time -> time - start_time, parsed_data[1]), parsed_data[2])
  
  # compute initial value
  initial_value = parsed_data[2][findall(x -> x == 0, parsed_data[1])[1]]

  # reshape into matrix
  initial_value, Array(transpose(hcat(parsed_data...)))
end

# ----- define data
u0, full_data = parse_data()
datasize = 500
times, ode_data = full_data[1,:][1:datasize], full_data[2,:][1:datasize]
tspan = (0.0, maximum(times))
tsteps = range(tspan[1], tspan[2], length = datasize)

# ----- define Neural ODE architecture
dudt2 = FastChain((x, p) -> FastDense(1, 50, tanh),
                            FastDense(50, 1))
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

##################### PLOTS: LOSSES ###############
scatter(losses, ylabel = "Loss",  yscale= :log, label = "Architecture1: 500 warmup, 500 sample")

################### RETRODICTED PLOTS: TIME SERIES #################
pl = scatter(tsteps, ode_data[1], color = :red, label = "CGM data", xlabel = "t", title = "CGM Neural ODE")

for k in 1:300
    resol = predict_neuralode(samples[100:end][rand(1:400)])
    plot!(tsteps,resol[1], alpha=0.04, color = :red, label = "")
end

idx = findmin(losses)[2]
prediction = predict_neuralode(samples[idx])

# plot!(tsteps, prediction[1,:], color = :black, w = 2, label = "")
# plot!(tsteps, prediction[2,:], color = :black, w = 2, label = "Best fit prediction", ylims = (-2.5, 3.5))

# ################### RETRODICTED PLOTS: TIME SERIES #################
# pl = scatter(tsteps, ode_data[1,:], color = :red, label = "Data: Var1", xlabel = "t", title = "Spiral Neural ODE")
# scatter!(tsteps, ode_data[2,:], color = :blue, label = "Data: Var2")

# for k in 1:300
#     resol = predict_neuralode(samples[100:end][rand(1:400)])
#     plot!(tsteps,resol[1,:], alpha=0.04, color = :red, label = "")
#     plot!(tsteps,resol[2,:], alpha=0.04, color = :blue, label = "")
# end

# idx = findmin(losses)[2]
# prediction = predict_neuralode(samples[idx])

# plot!(tsteps,prediction[1,:], color = :black, w = 2, label = "")
# plot!(tsteps,prediction[2,:], color = :black, w = 2, label = "Best fit prediction", ylims = (-2.5, 3.5))


# #################### RETRODICTED PLOTS - CONTOUR ####################
# pl = scatter(ode_data[1,:], ode_data[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Spiral Neural ODE")

# for k in 1:300
#     resol = predict_neuralode(samples[100:end][rand(1:400)])
#     plot!(resol[1,:],resol[2,:], alpha=0.04, color = :red, label = "")
# end

# plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Best fit prediction", ylims = (-2.5, 3))
