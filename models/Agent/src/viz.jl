using OpenStreetMapXPlot
using Plots
gr()

ac(agent) = agent.infected ? :green : :black
as(agent) = agent.infected ? 6 : 5

function plotagents(model)
    # Essentially a cut down version on plotabm
    ids = model.scheduler(model)
    colors = [ac(model[i]) for i in ids]
    sizes = [as(model[i]) for i in ids]
    markers = :circle
    pos = [osm_map_coordinates(model[i], model) for i in ids]

    scatter!(
        pos;
        markercolor = colors,
        markersize = sizes,
        markershapes = markers,
        label = "",
        markerstrokewidth = 0.5,
        markerstrokecolor = :black,
        markeralpha = 0.7,
    )
end

function make_anim(; n = 200, path = "outbreak.gif")
    model = initialise()

    frames = @animate for i in 0:n
        i > 0 && step!(model, agent_step!, 1)
        plotmap(model.space.m; width = 300*2, height = 200*2)
        plotagents(model)
    end

    gif(frames, path, fps = 15)
end

function make_anim_air(; n = 200, path = "outbreakair.gif")
    model = initialise_air()

    frames = @animate for i in 0:n
        i > 0 && step!(model, agent_step_air!, model_step_air!, 1)
        plotmap(model.space.m; width = 300*2, height = 200*2)
        plotagents(model)
    end

    gif(frames, path, fps = 15)
end

function simulate(; n = 200,
                    d = 50,
                    nagents = 100)
    model = initialise(; d = d,
                         nagents = nagents)

    for i in 0:n
        # @show i
        i > 0 && step!(model, agent_step!, 1)
    end
    model
end

function simulate_air(; n = 200, d = 50)
    model = initialise_air(; d = d)

    for i in 0:n
        # @show i
        i > 0 && step!(model, agent_step_air!, model_step_air!, 1)
    end
    model
end