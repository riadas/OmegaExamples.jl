using OrdinaryDiffEq, Flux, Random
using DiffEqFlux

include("data.jl")

function model(ode_data::AbstractArray, batch_size::Int=4, maxiters::Int=150, lr::Float64=0.01)
  # u0, ode_data = prepare_data("basis_steps", bin_size)
  u0 = ode_data[:, 1]
  data_dim = length(u0)
  datasize = length(ode_data[1,:])
  tspan = (0.0, Float64(datasize) - 1)
  tsteps = range(tspan[1], tspan[2], length = datasize)
  
  train_t = tsteps
  train_y = ode_data
  
  function neural_ode(t, data_dim; saveat = t)
      # println(data_dim)
      f = FastChain(FastDense(data_dim, 64, swish),
                    FastDense(64, 32, swish),
                    FastDense(32, data_dim))
  
      node = NeuralODE(f, (minimum(t), maximum(t)), Tsit5(),
                       saveat = saveat, abstol = 1e-9,
                       reltol = 1e-9)
  end
  
  function train_one_round(node, θ, y, opt, maxiters,
                           y0 = y[:, 1]; kwargs...)
  
      predict(θ) = Array(node(y0, θ))
      loss(θ) = begin
          pred = predict(θ)
          Flux.mse(pred, y)
      end
  
      θ = θ == nothing ? node.p : θ
      res = DiffEqFlux.sciml_train(
          loss, θ, opt,
          maxiters = maxiters;
          kwargs...
      )
      return res.minimizer
  end
  
  function train(θ = nothing, maxiters = maxiters, lr = lr)
      log_results(θs, losses) =
          (θ, loss) -> begin
              push!(θs, copy(θ))
              push!(losses, loss)
              false
          end
  
      θs, losses = [], []
      num_obs = batch_size:batch_size:length(train_t)
      for k in num_obs
        println(k)
        node = neural_ode(train_t[1:k], data_dim)
        θ = train_one_round(
            node, θ, train_y[:, 1:k],
            ADAMW(lr), maxiters;
            cb = log_results(θs, losses)
        )
      end
      θs, losses
  end
  
  # Random.seed!(1)
  θs, losses = train();
  
  idx = findmin(losses)[2]
  θ = θs[idx]
  y0 = train_y[:,1]
  node = neural_ode(train_t, data_dim)
  resol = Array(node(y0, θ))
  
  θs, losses, resol

  # v = "Steps"
  # pl = plot(tsteps, ode_data[1,:], color = :red, label = "Data: CGM", xlabel = "t", title = "CGM & $(v) & Bolus")
  # plot!(tsteps, ode_data[2,:], color = :blue, label = "Data: $(v)")
  # plot!(tsteps, ode_data[3,:], color = :green, label = "Data: Bolus")
  # plot!(tsteps,resol[1,:], alpha=0.5, color = :red, label = "CGM")
  # plot!(tsteps,resol[2,:], alpha=0.5, color = :blue, label = "Steps")
  # plot!(tsteps,resol[3,:], alpha=0.5, color = :green, label = "Bolus")
  # pl
end

function model_exo(non_exo_data::AbstractArray, exo_data::AbstractArray, batch_size::Int64=4, maxiters::Int=150, lr=0.01)
  # u0, ode_data = prepare_data("basis_steps", bin_size)
  u0 = non_exo_data[:, 1]
  datasize = length(ode_data[1,:])
  tspan = (0.0, Float64(datasize) - 1)
  tsteps = range(tspan[1], tspan[2], length = datasize)

  input_data_dim = size(non_exo_data, 1) + size(exo_data, 1)
  output_data_dim = size(non_exo_data, 1)
  
  train_t = tsteps
  train_y = ode_data
  
  function neural_ode(t, input_data_dim, output_data_dim; saveat = t)
      # println(data_dim)
      f = FastChain(FastDense(input_data_dim, 64, swish),
                    FastDense(64, 32, swish),
                    FastDense(32, output_data_dim))
      function f_exo(u, p)
        f(vcat(u[:,1], exo_data[:,1]), p)
      end
      node = NeuralODE(f_exo, (minimum(t), maximum(t)), Tsit5(),
                        saveat = saveat, abstol = 1e-9,
                        reltol = 1e-9)
  end
  
  function train_one_round(node, θ, y, opt, maxiters,
                            y0 = y[:, 1]; kwargs...)
      # println("y0")
      # println(size(y0))
      # println(y0)
  
      # println("theta")
      # println(size(node.p))
      # println(θ)
  
      predict(θ) = Array(node(y0, θ))
      loss(θ) = begin
          pred = predict(θ)
          # println("pred")
          # println(size(pred))
          # println(pred)
          # println(size(y)) 
          Flux.mse(pred, y)
      end
  
      θ = θ == nothing ? node.p : θ
      res = DiffEqFlux.sciml_train(
          loss, θ, opt,
          maxiters = maxiters;
          kwargs...
      )
      return res.minimizer
  end
  
  function train(θ = nothing, maxiters = maxiters, lr = lr)
      log_results(θs, losses) =
          (θ, loss) -> begin
              push!(θs, copy(θ))
              push!(losses, loss)
              false
          end
  
      θs, losses = [], []
      num_obs = batch_size:batch_size:length(train_t)
      for k in num_obs
        println(k)
        # println("train_y[:, 1:k]")
        # println(size(train_y[:, 1:k]))
        # println(train_y[:, 1:k])
        # println("train_t[1:k]")
        # println(size(train_t[1:k]))
        # println(train_t[1:k])
        node = neural_ode(train_t[1:k], input_data_dim, output_data_dim)
        θ = train_one_round(
            node, θ, train_y[:, 1:k],
            ADAMW(lr), maxiters;
            cb = log_results(θs, losses)
        )
      end
      θs, losses
  end
  
  # Random.seed!(1)
  θs, losses = train();
  
  idx = findmin(losses)[2]
  θ = θs[idx]
  y0 = train_y[:,1]
  node = neural_ode(train_t, data_dim)
  resol = Array(node(y0, θ))
  
  θs, losses, resol  
end

# for bin_size in [20, 15, 10, 5, 1]
#   for batch_size in [10, 8]
#     for maxiters in [150]
#       for lr in [0.01]
#         for i in 1:5
#           println("images/steps_bin_$(bin_size)_batch_$(batch_size)_maxIters_$(maxiters)_lr_$(lr)_i_$(i)") 
#           pl = model(bin_size, batch_size, maxiters, lr)
#           savefig(pl, "images/bolus_and_steps_images/bolus_steps_bin_$(bin_size)_batch_$(batch_size)_maxIters_$(maxiters)_lr_$(lr)_i_$(i).png")
#         end
#       end
#     end
#   end
# end