function layer_unpacker(i,l,k)
    if i == 1
        input_dim = 13
        output_dim = k[i]
    else
        input_dim = k[i-1]
        output_dim = k[i]
    end
    return input_dim, output_dim
end

function mse_scaled(y_pred,y_real)
    y_pred = StatsBase.reconstruct(dy,y_pred)
    y_real = StatsBase.reconstruct(dy,y_real)
    √(sum((y_pred .- y_real).^2))/length(y_real)
end

function mse_unscaled(y_pred,y_real)
    √(sum((y_pred .- y_real).^2))/length(y_real)
end

function likelihood_regression(x, y, iters)
    obs = obs_master;
    scores = []
    mses = []
    ls = []
    best_traces = []
    (best_trace,) = generate(interpolator, (x,), obs)
    best_score = get_score(best_trace)
    best_pred_y = transpose(G(x, best_trace))[:,1]
    best_mse = mse_unscaled(best_pred_y, y)
    
    (trace,) = generate(interpolator, (x,), obs)
    score = get_score(trace)
    pred_y = transpose(G(x, trace))[:,1]
    mse = mse_unscaled(pred_y, y)
    
    for i=1:iters
        (trace,) = generate(interpolator, (x,), obs)
        score = get_score(trace)
        pred_y = transpose(G(x, trace))[:,1]
        mse = mse_unscaled(pred_y, y)
        push!(scores,score)
        push!(mses,mse)
        if mse < best_mse
            best_mse = mse
            best_score = score
            best_trace = trace
            best_pred_y = pred_y
        end
    end
    return(best_trace, scores, mses)
end;