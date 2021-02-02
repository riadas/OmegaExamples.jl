include("data.jl")
include("model.jl")
include("model_bayes.jl")

# ----- START: OPTIMIZATION TEST FUNCTIONS ----- #

# glucose, steps, bolus, meals
function optim_all_vars_no_exo(bin_size::Int64, batch_size::Int=4, maxiters::Int=150, lr::Float64=0.01; hypo_id::Int64=1)
  u0, ode_data = prepare_all_data_meals_hypo(bin_size, hypo_id=hypo_id)
  samples, losses, best_pred = model(ode_data, batch_size, maxiters, lr)
  samples, losses, best_pred
end

# glucose, steps, bolus 
function optim_two_vars_no_exo(bin_size::Int64, batch_size::Int=4, maxiters::Int=150, lr::Float64=0.01)
  u0, ode_data = prepare_all_data(bin_size)
  samples, losses, best_pred = model(ode_data, batch_size, maxiters, lr)
  samples, losses, best_pred
end

# (glucose, steps) OR (glucose, bolus)
function optim_one_var_no_exo(var::String, bin_size::Int64, batch_size::Int=4, maxiters::Int=150, lr::Float64=0.01)
  u0, ode_data = prepare_data(var, bin_size)
  samples, losses, best_pred = model(ode_data, batch_size, maxiters, lr)
  samples, losses, best_pred
end

# glucose, steps, bolus, meals
function optim_all_vars_exo(bin_size::Int64, batch_size::Int=4, maxiters::Int=150, lr::Float64=0.01; hypo_id::Int64=1, log_dir="")
  u0, ode_data = prepare_all_data_meals_hypo(bin_size, hypo_id=hypo_id)
  non_exo_data = ode_data[1:2, :]
  exo_data = ode_data[3:4, :]
  # samples, losses, best_pred = model_exo(non_exo_data, exo_data, batch_size, maxiters, lr)
  model_exo(non_exo_data, exo_data, batch_size, maxiters, lr, log_dir=log_dir)
end

# glucose, steps, bolus 
function optim_two_vars_exo(bin_size::Int64, batch_size::Int=4, maxiters::Int=150, lr::Float64=0.01)
  u0, ode_data = prepare_all_data(bin_size)
  non_exo_data = ode_data[1:2, :]
  exo_data = ode_data[3, :]
  samples, losses, best_pred = model_exo(non_exo_data, exo_data, batch_size, maxiters, lr)
end

# (glucose, steps) OR (glucose, bolus)
function optim_one_var_exo(var::String, bin_size::Int64, batch_size::Int=4, maxiters::Int=150, lr::Float64=0.01)
  u0, ode_data = prepare_data(var, bin_size)
  if var == "basis_steps"
    samples, losses, best_pred = model(ode_data, batch_size, maxiters, lr)
  else
    non_exo_data = ode_data[1, :]
    exo_data = ode_data[2, :]
    samples, losses, best_pred = model_exo(non_exo_data, exo_data, batch_size, maxiters, lr)
  end
end

# ----- END: OPTIMIZATION TEST FUNCTIONS ----- #

# ----- START: NEURAL BAYES TEST FUNCTIONS ----- #

# glucose, steps, bolus, meals
function bayes_all_vars_no_exo(bin_size::Int64)
  u0, ode_data = prepare_all_data_meals_hypo(bin_size)
  samples, losses = model_bayes(ode_data)
  samples, losses
end
# glucose, steps, bolus 
function bayes_two_vars_no_exo(bin_size::Int64)
  u0, ode_data = prepare_all_data(bin_size)
  samples, losses = model_bayes(ode_data, batch_size, maxiters, lr)
  samples, losses
end

# (glucose, steps) OR (glucose, bolus)
function bayes_one_var_no_exo(var::String)
  u0, ode_data = prepare_data(var, bin_size)
  samples, losses = model_bayes(ode_data, batch_size, maxiters, lr)
  samples, losses
end

# glucose, steps, bolus, meals
function bayes_all_vars_exo(bin_size::Int64)
  u0, ode_data = prepare_all_data_meals_hypo(bin_size)
  non_exo_data = ode_data[1:2, :]
  exo_data = ode_data[3:4, :]
  samples, losses = model_bayes_exo(non_exo_data, exo_data, batch_size, maxiters, lr)
end

# glucose, steps, bolus 
function bayes_two_vars_exo(bin_size::Int64)
  u0, ode_data = prepare_all_data(bin_size)
  non_exo_data = ode_data[1:2, :]
  exo_data = ode_data[3, :]
  samples, losses = model_bayes_exo(non_exo_data, exo_data, batch_size, maxiters, lr)
end

# (glucose, steps) OR (glucose, bolus)
function bayes_one_var_exo(var::String)
  u0, ode_data = prepare_data(var, bin_size)
  if var == "basis_steps"
    samples, losses = model_bayes_exo(ode_data, batch_size, maxiters, lr)
  else
    non_exo_data = ode_data[1, :]
    exo_data = ode_data[2, :]
    samples, losses = model_bayes_exo(non_exo_data, exo_data, batch_size, maxiters, lr)
  end
end

# ----- END: NEURAL BAYES TEST FUNCTIONS ----- #