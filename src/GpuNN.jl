module GpuNN

function train(x, y, n_iters = 10, learning_rate = 0.01)
    input_size = size(x, 1)
    output_size = size(y, 1)

    params = _init_layers([input_size, 10, 5, output_size])
    nlayers = get_nlayers(params)

    nn_values

    for i = 1:n_iters
        nn_values = _forward(x, params)

        cost = _compute_cost(nn_values, y)
        println("Iteration $i cost: $cost")

        grads = _backward(params, nn_values, x, y)
        params = _update_weights(params, grads, learning_rate)
    end

    println("Train accuracy: $(accuracy(y, nn_values["A$nlayers"]))")
    nn_values
end

function accuracy(y, pred_proba)
    y_pred = mapslices(argmax, pred_proba, dims = 1)
    y_true = mapslices(argmax, y, dims = 1)

    return sum(y_true .== y_pred) / length(y)
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

        if i != nlayers
            relu!(nn_values["Z$i"])
        end
        nn_values["A$i"] = nn_values["Z$i"]

        last_output = nn_values["A$i"]
    end
    return nn_values
end

relu!(mat::Matrix) = begin
    mat[mat .< 0] .= 0
end

function _compute_cost(
    nn_values::Dict{String,Matrix{T}},
    y::Matrix{Int64},
)::Float64 where {T<:AbstractFloat}
    nlayers = get_nlayers(nn_values)
    y_pred = nn_values["A$nlayers"]
    cost = 1 / (2 * size(y, 2)) * sum((y_pred - y) .^ 2)
    return cost
end

function _backward(
    params::Dict{String,T},
    nn_values::Dict{String,T},
    x::T,
    y::Matrix{Int64},
)::Dict{String,T} where {T}
    nlayers = get_nlayers(params)
    nsamples = size(y, 2)
    grads = Dict()

    dA = 1 / nsamples * (nn_values["A$nlayers"] - y)
    dZ = dA

    for i = nlayers:-1:1
        if i != nlayers
            dA = transpose(params["W$(i + 1)"]) * dZ
            dZ::T = nn_values["A$i"] .>= 0
        end

        last_output = i == 1 ? x : nn_values["A$(i - 1)"]

        grads["W$i"] = 1 / nsamples * dZ * transpose(last_output)
        grads["B$i"] = 1 / nsamples * sum(dZ, dims = 2)
    end

    return grads
end

get_nlayers(params::Dict)::Int = (keys(params) |> length) / 2

function _update_weights(
    params::Dict{String,T},
    grads::Dict{String,T},
    learning_rate,
)::Dict{String,T} where {T}
    nlayers = get_nlayers(params)
    new_params = Dict()

    for i = 1:nlayers
        new_params["W$i"] = params["W$i"] - learning_rate * grads["W$i"]
        new_params["B$i"] = params["B$i"] - learning_rate * grads["B$i"]
    end

    return new_params
end

end
