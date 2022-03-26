module loss

function logistic(y_true, y_pred)::Float64
    return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
end

end
