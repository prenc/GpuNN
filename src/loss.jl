module loss

function categorical_crossentropy(y_true::Matrix, y_pred::Matrix)::Float64
    num_samples = size(y_true, 2)
    return -1 / num_samples *
           sum(y_true .* log.(y_pred) + (1 .- y_true) .* log.(1 .- y_pred))
end

end
