using Agents
using Distributions
using DataStructures
@agent Zombie OSMAgent begin
  infected::Bool
  history::CircularBuffer{Tuple{Int, Int}}
end

const somewherestart = (39.534773980413505, -119.78937575923226)
const somewherefinish = (39.52530416953533, -119.76949287425508)

const cambridgelb = (42.3664, -71.0736)
const cambridgeub = cambridgelb .+ 0.019

function initialise(; model = ABM(Zombie, OpenStreetMapSpace(map_path)),
                      start_ = somewherestart,
                      finish_ = somewherefinish,
                      map_path = TEST_MAP,
                      histlength = 5)
  

  for _ in 1:100
      start = random_position(model) # At an intersection
      finish = osm_random_road_position(model) # Somewhere on a road
      route = osm_plan_route(start, finish, model)
      add_agent!(start, model, route, finish, false, CircularBuffer{Tuple{Int, Int}}(histlength))
  end

  # We'll add patient zero at a specific (latitude, longitude)
  start = osm_road(start_, model)
  finish = osm_intersection(finish_, model)
  route = osm_plan_route(start, finish, model)
  add_agent!(start, model, route, finish, true, CircularBuffer{Tuple{Int, Int}}(histlength))
  return model
end

function agent_step!(agent, model)
  # Each agent will progress 25 meters along their route
  move_agent!(agent, model, 25)

  if osm_is_stationary(agent) && rand() < 0.1
      # When stationary, give the agent a 10% chance of going somewhere else
      osm_random_route!(agent, model)
      # Start on new route
      move_agent!(agent, model, 25)
  end

  if agent.infected
      # Agents will be infected if they get within 50 meters of a zombie.
      map(i -> model[i].infected = true, nearby_ids(agent, model, 50))
  end
end

bound(x, lb, ub) = max(lb, min(ub, x))

"Map lattitude and longitude to cell"
@inline function latlongtocell(lat, lon, nlat, nlon, latub, latlb, lonub, lonlb)
  cellΔlat = (latub - latlb) / nlat # TODO, only need to do this once 
  cellΔlon = (lonub - lonlb) / nlon # Likewie
  x = Int(floor((lat - latlb) / cellΔlat)) + 1
  y = Int(floor((lon - lonlb) / cellΔlon)) + 1

  return bound(x, 1, nlat), bound(y, 1, nlon)
end

function initialise_air(; start_ = somewherestart,
                          finish_ = somewherefinish,
                          map_path = TEST_MAP,
                          nlat = 100,
                          nlon = 100)
  grid = zeros(Float64, nlat, nlon)
  model_ = ABM(Zombie, OpenStreetMapSpace(map_path), properties = grid)
  model = initialise(; start_ = start_,
                       finish_ = finish_,
                       map_path = map_path,
                       model = model_)
  model
end

s(x, α = 1) = 1/ (1 + exp(-α * x))
s̃(x, α = 0.001) = 2(s(x, α) - 1/2 )

function model_step_air!(model)  
  grid = model.properties
  grid != 0.0
  for agent in allagents(model)
    for (x, y) in agent.history
      grid[x, y] += 1.0
    end
  end
end

@inline bounds(model) = model.space.m.bounds

function agent_step_air!(agent, model)
  grid = model.properties
  # Move as normal
  if osm_is_stationary(agent) && rand() < 0.1
      # When stationary, give the agent a 10% chance of going somewhere else
      osm_random_route!(agent, model)
      # Start on new route
      move_agent!(agent, model, 25)
  end

  # Update the grid with the history
  # grid != 0.0
  # for (x, y) in agent.history
  #   grid[x, y] += 1.0
  # end

  lat, lon = osm_latlon(agent, model)
  nlat, nlon = size(grid)

  b = bounds(model)
  lonlb = b.min_x
  lonub = b.max_x
  latlb = b.min_y
  latub = b.max_y

  xy = latlongtocell(lat, lon, nlat, nlon, latub, latlb, lonub, lonlb)
  push!(agent.history, xy)
  x, y = xy
  count = grid[x, y]
  p = s̃(count)
  if rand(Bernoulli(p))
    agent.infected = true
  end
end
