import Glucose
using Glucose: optim_all_vars_exo, prepare_all_data_meals_hypo
using SuParameters
using RunTools
using BSON
using Plots
using Flux, DiffEqFlux, OrdinaryDiffEq

function plot_best_model()
  Core.eval(Main, :(import NNlib))

  # data_directory = "/Users/riadas/Documents/urop/runs/odeoptim_firsttry"
  data_directory = "/scratch/riadas/runs/odeoptim_firsttry/rn0Gj1r_2021-02-02T09:20:14.067_sketch3"
  folder_names = readdir(data_directory)
  println(folder_names)
  output_dictionaries = []
  for folder in folder_names
    println("folder name")
    println(readdir(joinpath(data_directory, folder)))
    if "model.json" in readdir(joinpath(data_directory, folder)) # if run has finished
      d = BSON.load(joinpath(data_directory, folder, "model.bson"))
  
      id = split(folder, "_")[1]
      params = BSON.load(joinpath(data_directory, folder, "$(id).bson"))
      d[:hypoid] = params[:param][:hypoid]
      d[:binsize] = params[:param][:binsize]
  
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
  
  # plot
  # original data
  _, ode_data = prepare_all_data_meals_hypo(binsize, hypo_id=hypoid)
  tsteps = (0.0, 1.0, length=size(ode_data[1,:]))
  
  pl = plot(tsteps, ode_data[1,:], color = :red, label = "Data: CGM", xlabel = "t", title = "CGM, Steps, Bolus, Meals")
  plot!(tsteps, ode_data[2,:], color = :blue, label = "Data: Steps")
  plot!(tsteps, ode_data[3,:], color = :green, label = "Data: Bolus")
  plot!(tsteps, ode_data[4,:], color = :purple, label = "Data: Meals")
  plot!(tsteps,best_pred[1,:], alpha=0.3, color = :red, label = "CGM")
  plot!(tsteps,best_pred[2,:], alpha=0.3, color = :blue, label = "Steps")
  pl
end