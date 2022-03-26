module metric

function accuracy(y::Matrix, pred_proba::Matrix)
    y_pred = mapslices(argmax, pred_proba, dims = 1)
    y_true = mapslices(argmax, y, dims = 1)

    return sum(y_true .== y_pred) / length(y_true)
end

end
