# An Julia Implementation of No-U-Turn Sampler with Dual Averaging described in Algorithm 6 in Hoffman et al. (2011)
# Author: Kai Xu
# Date: 06/10/2016

function NUTS(trace, selection, ϵ, M, verbose=true)

  args = get_args(trace)
  retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
  argdiffs = map((_) -> NoChange(), args)
  (_, vals, gradient_trie) = choice_gradients(trace, selection, retval_grad)
  θ0 = to_array(vals, Float64)
    
  function L(θ)
    θtrace = from_array(vals, θ)
    (new_trace, _, _) = update(trace, args, argdiffs, θtrace)
    score = get_score(new_trace)
    return score
  end
    
  function ∇L(θ)
    θtrace = from_array(vals, θ)
    (new_trace, _, _) = update(trace, args, argdiffs, θtrace)
    (_, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    gradient = to_array(gradient_trie, Float64)
    #println(gradient)
    return gradient
  end 

  function leapfrog(θ, r, ϵ)

    r̃ = r + 0.5 * ϵ * ∇L(θ)
    θ̃ = θ + ϵ * r̃
    r̃ = r̃ + 0.5 * ϵ * ∇L(θ̃)
    return θ̃, r̃
  end

  function build_tree(θ, r, u, v, j, ϵ)

    if j == 0
      # Base case - take one leapfrog step in the direction v.
      θ′, r′ = leapfrog(θ, r, v * ϵ)
      n′ = u <= exp(L(θ′) - 0.5 * dot(r′, r′))
      s′ = u < exp(Δ_max + L(θ′) - 0.5 * dot(r′, r′))
      return θ′, r′, θ′, r′, θ′, n′, s′
    else
      # Recursion - build the left and right subtrees.
      θm, rm, θp, rp, θ′, n′, s′ = build_tree(θ, r, u, v, j - 1, ϵ)
      if s′ == 1
        if v == -1
          θm, rm, _, _, θ′′, n′′, s′′ = build_tree(θm, rm, u, v, j - 1, ϵ)
        else
          _, _, θp, rp, θ′′, n′′, s′′ = build_tree(θp, rp, u, v, j - 1, ϵ)
        end
        if rand() < n′′ / (n′ + n′′)
          θ′ = θ′′
        end
        s′ = s′′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
        n′ = n′ + n′′
      end
      return θm, rm, θp, rp, θ′, n′, s′
    end
  end

  #∇L = θ -> ForwardDiff.gradient(L, θ)  # generate gradient function

  θs = [zeros(length(θ0)) for i=1:M+1]
  θs[1] = θ0

  if verbose
    println("[eff_NUTS] start sampling for $M samples with ϵ=$ϵ")
  end

  for m = 1:M
    #r0 = randn(length(θ0))
    r_mean = zeros(length(θ0))
    r_cov = Diagonal([5.0 for j=1:length(r_mean)])
    r0 = mvnormal(r_mean,r_cov)
    u = rand() * exp(L(θs[m]) - 0.5 * dot(r0, r0)) # Note: θ^{m-1} in the paper corresponds to
                                                   #       `θs[m]` in the code
    θm, θp, rm, rp, j, θs[m + 1], n, s = θs[m], θs[m], r0, r0, 0, θs[m], 1, 1
    while s == 1
      v_j = rand([-1, 1]) # Note: this variable actually does not depend on j;
                          #       it is set as `v_j` just to be consistent to the paper
      if v_j == -1
        θm, rm, _, _, θ′, n′, s′ = build_tree(θm, rm, u, v_j, j, ϵ)
      else
        _, _, θp, rp, θ′, n′, s′ = build_tree(θp, rp, u, v_j, j, ϵ)
      end
      if s′ == 1
        if rand() < min(1, n′ / n)
          θs[m + 1] = θ′
        end
      end
      n = n + n′
      s = s′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
      j = j + 1
    end
  end

  if verbose
    println()
    println("[eff_NUTS] sampling complete")
  end

  traces = []
  for i = 1:length(θs)
    θ = from_array(vals, θs[i])
    #println(θ)
    (trace, _, _) = update(trace, args, argdiffs, θ)
    push!(traces, trace)
  end
    
  return traces
end