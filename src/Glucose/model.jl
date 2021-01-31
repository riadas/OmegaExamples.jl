using OrdinaryDiffEq, Flux, Random
using DiffEqFlux



function neural_ode(t, data_dim; saveat = t)
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
        ŷ = predict(θ)
        Flux.mse(ŷ, y)
    end

    θ = θ == nothing ? node.p : θ
    res = DiffEqFlux.sciml_train(
        loss, θ, opt,
        maxiters = maxiters;
        kwargs...
    )
    return res.minimizer
end

function train(θ = nothing, maxiters = 150, lr = 1e-2)
    log_results(θs, losses) =
        (θ, loss) -> begin
            push!(θs, copy(θ))
            push!(losses, loss)
            false
        end

    θs, losses = [], []
    num_obs = 4:4:length(train_t)
    for k in num_obs
        node = neural_ode(train_t[1:k], size(y, 1))
        θ = train_one_round(
            node, θ, train_y[:, 1:k],
            ADAMW(lr), maxiters;
            cb = log_results(θs, losses)
        )
    end
    θs, losses
end



Random.seed!(1)
θs, losses = train();