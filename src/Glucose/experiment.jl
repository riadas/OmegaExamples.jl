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
      # println(data_name)
      # println(event)
       # time in minutes
      if (data_name in ["glucose_level", "basis_bolus"])
        push!(parsed_data[1], ceil(DateTime(attribute(XMLElement(event), "ts"), "dd-mm-yyyy HH:MM:SS"), Dates.Minute(5)).instant.periods.value/(1000.0 * 60))
        push!(parsed_data[2], parse(Float64, attribute(XMLElement(event), "value")))
      elseif (data_name == "meal")
        push!(parsed_data[1], ceil(DateTime(attribute(XMLElement(event), "ts"), "dd-mm-yyyy HH:MM:SS"), Dates.Minute(5)).instant.periods.value/(1000.0 * 60))
        push!(parsed_data[2], parse(Float64, attribute(XMLElement(event), "carbs")))
      elseif (data_name == "bolus")
        # if (attribute(XMLElement(event), "type") != "normal")
        #   println("NOT NORMAL")
        # end

        # if (attribute(XMLElement(event), "ts_begin") != attribute(XMLElement(event), "ts_end"))
        #   println("NOT INSTANTANEOUS")
        # end
        # println("hello")
        # println(attribute(XMLElement(event), "ts_begin"))
        # println(attribute(XMLElement(event), "ts_end"))
        push!(parsed_data[1], ceil(DateTime(attribute(XMLElement(event), "ts_begin"), "dd-mm-yyyy HH:MM:SS"), Dates.Minute(5)).instant.periods.value/(1000.0 * 60))
        push!(parsed_data[2], parse(Float64, attribute(XMLElement(event), "dose")))
      end
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
_, full_data_glucose = parse_data()
_, full_data_bolus = parse_data(data_name="bolus") # basis_bolus

times_glucose, ode_data_glucose = full_data_glucose[1,:], full_data_glucose[2,:]
times_bolus, ode_data_bolus = full_data_bolus[1,:], full_data_bolus[2,:]

times = times_glucose[5183:5999]
start_time = minimum(times)
times = map(time -> time - start_time, times)

println(string("start_time: ", start_time))
println(string("length(times): ", length(times)))
println(times)

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

ode_data_bolus = ode_data_bolus_new .* 50.0

# println(ode_data_glucose)
# println(ode_data_bolus)

println(length(ode_data_glucose))
println(length(ode_data_bolus))

start_time_idx = findall(time -> time == 0, times)[1]
u0 = [ode_data_glucose[start_time_idx]; ode_data_bolus[start_time_idx]]
ode_data = Array(transpose(hcat([ode_data_glucose, ode_data_bolus]...)))

datasize = length(times)
tspan = (0.0, maximum(times))
tsteps = range(tspan[1], tspan[2], length = datasize)

pl = scatter(tsteps, ode_data[1,:], color = :red, label = "Data: CGM", xlabel = "t", title = "CGM & Basis bolus")
scatter!(tsteps, ode_data[2,:], color = :blue, label = "Data: Basis bolus")

# savefig(pl, "glucose_bolus.png")