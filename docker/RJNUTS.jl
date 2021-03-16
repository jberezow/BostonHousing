function load_layer(l)
    layer_selection = select()
    push!(layer_selection, (:W,l))
    push!(layer_selection, (:b,l))
    return layer_selection
end

@gen function gibbs_hyperparameters(trace)
    obs_new = choicemap()::ChoiceMap
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    
    for i=1:trace[:l] + 1
        #Biases
        bias = trace[(:b,i)]
        
        n = length(bias)
        α = α₁ + (n/2)
        
        Σ = sum(bias.^2)/2 
        β = 1/(1/β₁ + Σ)
        
        τᵦ ~ gamma(α,β)
        
        #Weights
        i == 1 ? α₀ = α₁ : α₀ = α₂
        i == 1 ? β₀ = β₁ : β₀ = β₂
        
        weight = trace[(:W,i)]
        
        n = length(weight)
        α = α₀ + (n/2)
        
        Σ = sum(weight.^2)/2
        β = 1/(1/β₀ + Σ)
        
        τ ~ gamma(α,β)
        
        obs_new[(:τ,i)] = τ
        obs_new[(:τᵦ,i)] = τᵦ
    end
    
    (new_trace,_,_,_) = update(trace, args, argdiffs, obs_new)
    
    return new_trace
end

function G2(x, trace)
    activation = relu
    l = trace[:l]
    ks = [trace[(:k,i)] for i=1:l]

    for i=1:l
        in_dim, out_dim = layer_unpacker(i, l, ks)
        W = reshape(trace[(:W,i)], out_dim, in_dim)
        b = reshape(trace[(:b,i)], trace[(:k,i)])
        nn = Dense(W, b, activation)
        x = nn(x)
    end

    Wₒ = reshape(trace[(:W,l+1)], 1, ks[l])
    bₒ = reshape(trace[(:b,l+1)], 1)

    nn_out = Dense(Wₒ, bₒ)
    return nn_out(x)

end;

@gen function gibbs_noise(trace)
    
    obs_new = choicemap()::ChoiceMap
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    
    n = length(trace[:y])
    α = αᵧ + (n/2)
    
    x = get_args(trace)[1]
    y_pred = transpose(G2(x,trace))[:,1]
    y_real = trace[:y]
    Σᵧ = sum((y_pred .- y_real).^2)/2
    β = 1/(1/βᵧ + Σᵧ)
    
    τ ~ gamma(α,β)
    obs_new[:τᵧ] = τ
    
    (new_trace,_,_,_) = update(trace, args, argdiffs, obs_new)
    
    return new_trace
end

function nuts_parameters(trace)
    
    l = trace[:l]
    param_selection = select()
    for i=1:l+1 #Number of Layers
        push!(param_selection, (:W,i))
        push!(param_selection, (:b,i))
    end
    
    prev_score = get_score(trace)
    
    acc = 0
    for i=1:iters
        new_trace = NUTS(trace, param_selection, acc_prob, m, m, false)[m+1]
        new_score = get_score(new_trace)
        if prev_score != new_score
            return (new_trace, 1)
        else
            return (trace, 0)
        end
    end
    
    return (trace, acc)
end

function layer_nuts(trace,mode="draw")
    prev_score = get_score(trace)
    new_trace = trace
    if mode == "draw"
        mode = bernoulli(0.5) ? "for" : "back"
    end
    
    #Backward Pass
    if mode == "back"
        for j=1:new_trace[:l]+1
            v = new_trace[:l]+2 - j
            layer_selection = load_layer(v)
            #println("Current Layer: $v")
            new_trace = NUTS(new_trace, layer_selection, acc_prob, m, m, false)[m+1]
        end
        
    #Forward Pass
    else
        for j=1:new_trace[:l]+1
            v = j
            layer_selection = load_layer(v)
            #println("Current Layer: $v")
            new_trace = NUTS(new_trace, layer_selection, acc_prob, m, m, false)[m+1]
        end
    end
    
    new_score = get_score(new_trace)
    if prev_score != new_score
        return (new_trace, 1)
    else
        return (trace, 0)
    end
    
end

function layer_parameter(trace)
    obs = obs_master
    for i=1:trace[:l]+1
        obs[(:τ,i)] = trace[(:τ,i)]
        obs[(:τᵦ,i)] = trace[(:τᵦ,i)]
    end
    obs[:τᵧ] = trace[:τᵧ]
    
    init_trace = trace
    
    #################################################RJNUTS#################################################
    #NUTS Step 1
    trace_tilde = trace
    for i=1:1
        (trace_tilde,) = layer_nuts(trace_tilde,"for")
    end
    
    #Reversible Jump Step
    (trace_prime, q_weight) = layer_change(trace_tilde)
    
    #NUTS Step 2
    trace_star = trace_prime
    for i=1:1
        (trace_star,) = layer_nuts(trace_star,"back")
    end
    #################################################RJNUTS#################################################
        
    model_score = -get_score(init_trace) + get_score(trace_star)
    across_score = model_score + q_weight

    if rand() < exp(across_score)
        println("********** Accepted: $(trace_star[:l]) **********")
        return (trace_star, 1)
    else
        return (init_trace, 0)
    end
end

function RJNUTS(trace, iters=ITERS, chain=CHAINS)
    traces = []
    scores = []
    across_acceptance = []
    within_acceptance = []
    
    for i=1:iters
        (trace, accepted) = layer_parameter(trace)
        push!(across_acceptance, accepted)
        trace  = gibbs_hyperparameters(trace)
        trace  = gibbs_noise(trace)
        (trace, accepted)  = layer_nuts(trace)
        push!(within_acceptance, accepted)
        push!(scores,get_score(trace))
        push!(traces, trace)
        println("$i : $(get_score(trace))")
        if i%5 == 0
            a_acc = 100*(sum(across_acceptance)/length(across_acceptance))
            w_acc = 100*(sum(within_acceptance)/length(within_acceptance))
            println("Epoch $i A Acceptance Probability: $a_acc %")
            println("Epoch $i W Acceptance Probability: $w_acc %")
        end
    end
    
    return traces, scores
end
