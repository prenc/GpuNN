module utils

function to_categorical(T::Type, a::Vector, num_classes)::Matrix{T}
    num_instances = length(a)
    cat_a = zeros(T, num_classes, num_instances)

    for (class, col) in zip(a, eachcol(cat_a))
        col[class + 1] = 1  # stupid as hell, needs to be reworked
    end

    return cat_a
end

to_categorical(a, num_classes) = to_categorical(Float32, a, num_classes)

end
