module loss

function logistic(y_true, y_pred)::Float64
    y_true == y_pred && return 0
    y_true != y_pred && y_pred in (0, 1) && return Inf

    return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
end

end
