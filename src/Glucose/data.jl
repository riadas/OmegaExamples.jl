using LightXML
using Dates
using Statistics 
using Plots

FILE_LOCATION = "../../../OhioT1DM-training/559-ws-training.xml"

function average_over_bins(data::AbstractArray, bin_size::Int64)
  size = min(length(data), bin_size)
  bins = Iterators.partition(data, size) |> collect
  map(row -> mean(row), bins)
end

function normalize(data::AbstractArray)
  (data .- mean(data)) ./ std(data)
end

function normalize_binned_data(data::AbstractArray, bin_size::Int64)
  normalize(average_over_bins(data, bin_size))
end

function prepare_data(var::String, bin_size::Int64; file_location::String=FILE_LOCATION)
  if var == "basis_steps"
    ode_data = prepare_steps_data(file_location=file_location)
  elseif var == "bolus"
    ode_data = prepare_bolus_data(file_location=file_location)
  end

  println(length(ode_data[1, :]))
  println(length(ode_data[2, :]))

  glucose_data = normalize_binned_data(ode_data[1,:], bin_size)
  other_data = normalize_binned_data(ode_data[2,:], bin_size)

  println(length(glucose_data))
  println(length(other_data))

  u0 = [glucose_data[1]; other_data[1]]

  (u0, Array(transpose(hcat([glucose_data, other_data]...))))
end

function prepare_steps_data(;file_location::String=FILE_LOCATION)
  full_data_glucose = parse_data_from_xml("glucose_level"; round_time=true, file_location=file_location)
  full_data_steps = parse_data_from_xml("basis_steps"; round_time=true, file_location=file_location)
    
  times_glucose, ode_data_glucose = full_data_glucose[1,:], full_data_glucose[2,:] # 1000 is arbitrary
  times_steps, ode_data_steps = full_data_steps[1,:], full_data_steps[2,:] # 1000 is arbitrary
  
  sorted_segments = sorted_glucose_steps_segments()
  times = sorted_segments[1]
  start_time = minimum(times)
  times = map(time -> time - start_time, times)
    
  ode_data_glucose = map(i -> ode_data_glucose[i], findall(time -> (time - start_time) in times, unique(times_glucose)))
  ode_data_steps = map(i -> ode_data_steps[i], findall(time -> (time - start_time) in times, unique(times_steps)))
  
  Array(transpose(hcat([ode_data_glucose, ode_data_steps]...)))
end

function prepare_all_data(bin_size::Int64;file_location::String=FILE_LOCATION)
  full_data_glucose = parse_data_from_xml("glucose_level"; round_time=true, file_location=file_location)
  full_data_bolus = parse_data_from_xml("bolus"; round_time=true, file_location=file_location)
  full_data_steps = parse_data_from_xml("basis_steps"; round_time=true, file_location=file_location)

  times_glucose, ode_data_glucose = full_data_glucose[1,:], full_data_glucose[2,:]
  times_bolus, ode_data_bolus = full_data_bolus[1,:], full_data_bolus[2,:]
  times_steps, ode_data_steps = full_data_steps[1,:], full_data_steps[2,:] # 1000 is arbitrary
  
  sorted_segments = sorted_glucose_steps_segments()

  sorted_segment_intersections = map(segment -> length(intersect(segment, times_bolus)), sorted_segments)
  idx = findmax(sorted_segment_intersections)[2]
  times = sorted_segments[idx]

  start_time = minimum(times)
  times = map(time -> time - start_time, times)

  ode_data_glucose = normalize_binned_data(map(i -> ode_data_glucose[i], findall(time -> (time - start_time) in times, times_glucose)), bin_size)
  ode_data_steps = normalize_binned_data(map(i -> ode_data_steps[i], findall(time -> (time - start_time) in times, times_steps)), bin_size)

  ode_data_bolus_new = []
  for time in times
    if (time + start_time) in times_bolus
      idx = findall(x -> x == (time + start_time), unique(times_bolus))[1]
      push!(ode_data_bolus_new, ode_data_bolus[idx])
    else
      push!(ode_data_bolus_new, 0.0)
    end
  end
  ode_data_bolus = normalize_binned_data(ode_data_bolus_new, bin_size)

  ode_data = Array(transpose(hcat([ode_data_glucose, ode_data_steps, ode_data_bolus]...)))
  u0 = ode_data[:, 1]

  u0, ode_data
end

function prepare_bolus_data(;file_location::String=FILE_LOCATION)
  full_data_glucose = parse_data_from_xml("glucose_level"; round_time=true, file_location=file_location)
  full_data_bolus = parse_data_from_xml("bolus"; round_time=true, file_location=file_location)

  times_glucose, ode_data_glucose = full_data_glucose[1,:], full_data_glucose[2,:]
  times_bolus, ode_data_bolus = full_data_bolus[1,:], full_data_bolus[2,:]

  times = times_glucose[5183:5999]
  start_time = minimum(times)
  times = map(time -> time - start_time, times)

  ode_data_glucose = ode_data_glucose[5183:5999]
  ode_data_bolus_new = []

  for time in times
    if (time + start_time) in times_bolus
      idx = findall(x -> x == (time + start_time), unique(times_bolus))[1]
      push!(ode_data_bolus_new, ode_data_bolus[idx])
    else
      push!(ode_data_bolus_new, 0.0)
    end
  end

  ode_data_bolus = ode_data_bolus_new

  println(length(ode_data_glucose))
  println(length(ode_data_bolus))

  Array(transpose(hcat([ode_data_glucose, ode_data_bolus]...)))

end

function sorted_glucose_steps_segments(;file_location=FILE_LOCATION)
  full_data_glucose = parse_data_from_xml("glucose_level"; round_time=true, file_location=file_location)
  full_data_bolus = parse_data_from_xml("bolus"; round_time=true, file_location=file_location)
  full_data_steps = parse_data_from_xml("basis_steps"; round_time=true, file_location=file_location)

  times_glucose, ode_data_glucose = full_data_glucose[1,:], full_data_glucose[2,:]
  times_bolus, ode_data_bolus = full_data_bolus[1,:], full_data_bolus[2,:]
  times_steps, ode_data_steps = full_data_steps[1,:], full_data_steps[2,:] # 1000 is arbitrary

  times_gs = intersect(times_glucose, times_steps)
  shifted_times_gs = [0.0, times_gs[1:length(times_gs) - 1]...][1:length(times_gs)]
  consec_time_differences = times_gs .- shifted_times_gs
  jump_indices = findall(x -> !(x in [0.0, 5.0]), consec_time_differences)
  shifted_jump_indices = [0.0, jump_indices[1:length(jump_indices) - 1]...][1:length(jump_indices)]
  consec_segment_lengths = map(x -> Int(x), jump_indices .- shifted_jump_indices)

  segments = []
  for i in 1:length(jump_indices)
    if i == 1
      push!(segments, times_gs[1:jump_indices[i] - 1])
    else
      push!(segments, times_gs[jump_indices[i - 1]:jump_indices[i] - 1])
    end
  end

  reverse(sort(segments, by=length))
end

function parse_data_from_xml(data_name::String; round_time::Bool=false, file_location::String=FILE_LOCATION)
  # xdoc is an instance of XMLDocument, which maintains a tree structure
  xdoc = parse_file(file_location)

  # get the root element
  xroot = root(xdoc)  # an instance of XMLElement

  data_element = find_element(xroot, data_name)

  parsed_data = [[],[]]

  for event in child_nodes(data_element)
    if is_elementnode(event)
      # println(event)
      if (data_name in ["glucose_level", "basis_steps"])
        push!(parsed_data[1], ceil(DateTime(attribute(XMLElement(event), "ts"), "dd-mm-yyyy HH:MM:SS"), Dates.Minute(round_time ? 5 : 1)).instant.periods.value/(1000.0 * 60))
        push!(parsed_data[2], parse(Float64, attribute(XMLElement(event), "value")))
      elseif (data_name == "meal")
        push!(parsed_data[1], ceil(DateTime(attribute(XMLElement(event), "ts"), "dd-mm-yyyy HH:MM:SS"), Dates.Minute(round_time ? 5 : 1)).instant.periods.value/(1000.0 * 60))
        push!(parsed_data[2], parse(Float64, attribute(XMLElement(event), "carbs")))
      elseif (data_name == "bolus")
        push!(parsed_data[1], ceil(DateTime(attribute(XMLElement(event), "ts_begin"), "dd-mm-yyyy HH:MM:SS"), Dates.Minute(round_time ? 5 : 1)).instant.periods.value/(1000.0 * 60))
        push!(parsed_data[2], parse(Float64, attribute(XMLElement(event), "dose")))
      end
    end
  end

  free(xdoc)

  println(string("length(parsed_data): ", length(parsed_data[1])))
  
  # shift times to begin at t=0
  start_time = minimum(parsed_data[1])
  parsed_data = (map(time -> time - start_time, parsed_data[1]), parsed_data[2])
  
  # reshape into matrix
  Array(transpose(hcat(parsed_data...)))
end

function plot_data(ode_data::AbstractArray, var1::String, var2::String)
  tspan = (0.0, Float64(length(ode_data[1,:])-1))
  tsteps = range(tspan[1], tspan[2], length = length(ode_data[1,:]))

  pl = plot(tsteps, ode_data[1,:], color = :red, label = "Data: $(var1)", xlabel = "t", title = "$(var1) and $(var2)")
  plot!(tsteps, ode_data[2,:], color = :blue, label = "Data: $(var2)")
  pl
end

function plot_data_all(ode_data::AbstractArray, var1::String, var2::String)
  tspan = (0.0, Float64(length(ode_data[1,:])-1))
  tsteps = range(tspan[1], tspan[2], length = length(ode_data[1,:]))

  pl = plot(tsteps, ode_data[1,:], color = :red, label = "Data: Glucose", xlabel = "t", title = "Glucose, $(var1), and $(var2)")
  plot!(tsteps, ode_data[2,:], color = :blue, label = "Data: $(var1)")
  plot!(tsteps, ode_data[3,:], color = :green, label = "Data: $(var2)")
  pl
end