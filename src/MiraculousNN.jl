module MiraculousNN

export fit

using ProgressMeter

include("metric.jl")
include("forprop.jl")
include("backprop.jl")
include("loss.jl")
include("utils.jl")

using .metric
using .forprop
using .backprop
using .loss
using .utils


function fit(x, y; epochs = 10, batch_size = 32, learning_rate = 0.1, verbose = true)
    input_size = size(x, 1)
    output_size = size(y, 1)

    nn_params = _init_layers([input_size, 512, 512, output_size])
    nlayers = get_nlayers(nn_params)

    nn_values = Dict()
    grads = Dict()

    num_batches = div(size(x, 2), batch_size) - 1

    for i = 1:epochs
        p = Progress(
            num_batches;
            desc = "Epoch $i",
            barglyphs = BarGlyphs("|-> |"),
            enabled = verbose,
        )

        for b_idx = 1:num_batches
            b = b_idx * batch_size
            x_batch = x[:, b:(b + batch_size - 1)]
            y_batch = y[:, b:(b + batch_size - 1)]

            nn_values = _forward(x_batch, nn_params)

            grads = _backward(nn_params, nn_values, x_batch, y_batch)
            _update_weights!(nn_params, grads, learning_rate)

            nn_values = _forward(x, nn_params)
            pred_proba = nn_values["A$nlayers"]

            cost = _compute_cost(y, pred_proba)
            train_accuracy = metric.accuracy(y, pred_proba)

            ProgressMeter.next!(
                p;
                showvalues = [(:loss, cost), (:train_accuracy, train_accuracy)],
            )
        end
    end
end


function _init_layers(layer_sizes)::Dict{String,Matrix{Float64}}
    params = Dict()

    for i = 2:length(layer_sizes)
        params["W$(i - 1)"] = rand(Float64, layer_sizes[i], layer_sizes[i - 1]) * 0.01
        params["B$(i - 1)"] = zeros(Float64, layer_sizes[i], 1)
    end
    return params
end


function _forward(x::T, params::Dict{String,T})::Dict where {T}
    nlayers = get_nlayers(params)
    nn_values = Dict{String,T}()

    last_output = x

    for i = 1:nlayers
        nn_values["Z$i"] = params["W$i"] * last_output .+ params["B$i"]

        activation = i == nlayers ? forprop.softmax : forprop.relu
        nn_values["A$i"] = activation(nn_values["Z$i"])

        last_output = nn_values["A$i"]
    end
    return nn_values
end


function _backward(
    params::Dict{String,T},
    nn_values::Dict{String,T},
    x::T,
    y::Matrix{Int64},
)::Dict{String,T} where {T}
    nlayers = get_nlayers(params)
    num_samples = size(y, 2)
    grads = Dict()

    dA = nn_values["A$nlayers"]
    dZ = backprop.softmax.(dA, y)

    for i = nlayers:-1:1
        if i != nlayers
            dA = transpose(params["W$(i + 1)"]) * dZ
            dZ = backprop.relu.(dA, nn_values["A$i"])
        end

        next_output = i == 1 ? x : nn_values["A$(i - 1)"]

        grads["W$i"] = dZ * transpose(next_output) / num_samples 
        grads["B$i"] = sum(dZ, dims = 2) / num_samples 
    end

    return grads
end

get_nlayers(params::Dict)::Int = (keys(params) |> length) / 2

function _update_weights!(
    params::Dict{String,T},
    grads::Dict{String,T},
    learning_rate,
)::Nothing where {T}
    nlayers = get_nlayers(params)

    for i = 1:nlayers
        params["W$i"] .-= learning_rate * grads["W$i"]
        params["B$i"] .-= learning_rate * grads["B$i"]
    end
end

function _compute_cost(y, pred_proba)
    num_samples = size(y, 2)
    return sum(loss.logistic.(y, pred_proba)) / num_samples
end


end
