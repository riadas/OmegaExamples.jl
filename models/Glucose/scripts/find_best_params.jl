import Glucose
using Glucose: optim_all_vars_exo, prepare_all_data_meals_hypo
using SuParameters
using RunTools
using BSON
using Plots
using Flux, DiffEqFlux, OrdinaryDiffEq

function plot_best_model()
  Core.eval(Main, :(import NNlib))

  data_directory = "/Users/riadas/Documents/urop/OmegaExamples.jl/results/optim_runs_exo/"
  # data_directory = "/scratch/riadas/runs/odeoptim_firsttry/"
  folder_names = readdir(data_directory)
  println(folder_names)
  output_dictionaries = []
  for folder in folder_names
    println("folder name")
    println(readdir(joinpath(data_directory, folder)))
    if "model.bson" in readdir(joinpath(data_directory, folder)) # if run has finished
      println("hello")
      d = BSON.load(joinpath(data_directory, folder, "model.bson"))
  
      id = split(folder, "_")[1]
      params = BSON.load(joinpath(data_directory, folder, "$(id).bson"))
      d[:hypoid] = params[:param][:hypoid]
      d[:binsize] = params[:param][:binsize]
      d[:batchsize] = params[:param][:batchsize]
  
      push!(output_dictionaries, d)
  
    end
  end
  
  sorted_dictionaries = sort(output_dictionaries, by=(x -> x[:loss]))
  optimal_dict = sorted_dictionaries[1]
  
  node = optimal_dict[:nn_model]
  theta = optimal_dict[:theta] 
  loss = optimal_dict[:loss]
  best_pred = optimal_dict[:best_pred]
  y0 = optimal_dict[:y0]
  hypoid = optimal_dict[:hypoid]
  binsize = optimal_dict[:binsize]
  batchsize = optimal_dict[:batchsize]

  # plot
  # original data

  for i in 1:length(sorted_dictionaries)
    optimal_dict = sorted_dictionaries[i]
    loss = optimal_dict[:loss]
    best_pred = optimal_dict[:best_pred]
    y0 = optimal_dict[:y0]
    hypoid = optimal_dict[:hypoid]
    binsize = optimal_dict[:binsize]
    batchsize = optimal_dict[:batchsize]

    _, ode_data = prepare_all_data_meals_hypo(binsize, hypo_id=hypoid)
    tsteps = range(0.0, 1.0, length=length(ode_data[1,:]))

    pl = plot(tsteps, ode_data[1,:], color = :red, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals, loss=$(loss))")
    plot!(tsteps, ode_data[2,:], color = :blue, label = "Data: Steps")
    plot!(tsteps, ode_data[3,:], color = :green, label = "Data: Bolus")
    plot!(tsteps, ode_data[4,:], color = :purple, label = "Data: Meals")
    plot!(tsteps, best_pred[1,:], alpha=0.3, color = :red, label = "CGM")
    plot!(tsteps, best_pred[2,:], alpha=0.3, color = :blue, label = "Steps")
    savefig(pl, "optim_plots/optim_bin_$(binsize)_batch_$(batchsize)_hypoid_$(hypoid)_i_$(i).png")
  end
  
  pl = plot(tsteps, ode_data[1,:], color = :red, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals")
  plot!(tsteps, ode_data[2,:], color = :blue, label = "Data: Steps")
  plot!(tsteps, ode_data[3,:], color = :green, label = "Data: Bolus")
  plot!(tsteps, ode_data[4,:], color = :purple, label = "Data: Meals")
  plot!(tsteps, best_pred[1,:], alpha=0.3, color = :red, label = "CGM")
  plot!(tsteps, best_pred[2,:], alpha=0.3, color = :blue, label = "Steps")
  pl, node, theta, loss, best_pred, y0, hypoid, binsize, batchsize
end

function find_model(index::Int)
  Core.eval(Main, :(import NNlib))

  data_directory = "/Users/riadas/Documents/urop/OmegaExamples.jl/results/optim_runs_exo/"
  # data_directory = "/scratch/riadas/runs/odeoptim_firsttry/"
  folder_names = readdir(data_directory)
  println(folder_names)
  output_dictionaries = []
  for folder in folder_names
    if "model.bson" in readdir(joinpath(data_directory, folder)) # if run has finished
      d = BSON.load(joinpath(data_directory, folder, "model.bson"))
  
      # id = split(folder, "_")[1]
      # params = BSON.load(joinpath(data_directory, folder, "$(id).bson"))
      # d[:hypoid] = params[:param][:hypoid]
      # d[:binsize] = params[:param][:binsize]
      # d[:batchsize] = params[:param][:batchsize]
  
      push!(output_dictionaries, d)
  
    end
  end
  
  sorted_dictionaries = sort(output_dictionaries, by=(x -> x[:loss]))
  sorted_dictionaries[index][:nn_model]
  end
end