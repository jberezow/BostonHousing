current_dir = pwd()
app_dir = "/app"
cd(app_dir)

push!(LOAD_PATH, app_dir)
push!(LOAD_PATH, current_dir)

using Gen
using Distributions
using LinearAlgebra
using Flux
using Random
using Distances
using JLD
using Serialization
using StatsBase
using BNN

include("NUTS.jl")
include("RJNUTS.jl")
include("utils.jl")
include("proposals.jl")
include("LoadData.jl");

println("Packages Loaded")

filename = "Parallel1.jld"
ITERS = 10
CHAINS = Threads.nthreads()
println("Number of threads: $CHAINS")

#---------------
#Hyperparameters
#---------------

#NUTS hyperparameters
Δ_max = 10
acc_prob = 0.65
m=1

#Network hyperparameters
k_real = 4 #Number of hidden nodes per layer
k_vector = [0.0 for i=1:k_real]
k_vector[k_real] = 1.0

#Layer hyperparameters
l_range = 8 #Maximum number of layers in the network
l_list = [Int(i) for i in 1:l_range]
l_real = 1;

#Hyperprior Hyperparameters
αᵧ = 1 #Regression Noise Shape
βᵧ = 1 #Regression Noise Scale/Rate
α₁ = 1 #Input Weights, Biases Shape
β₁ = 1 #Input Weights, Biases Scale/Rate
α₂ = 1 #Hidden & Output Weights Shape
β₂ = k_real; #Hidden & Output Weights Scale

obs_master = choicemap()::ChoiceMap
obs_master[:y] = y_train
obs = obs_master;

#-----------
#Parallelize
#-----------

println("Initializing Traces")
println("------------------")

starting_traces = [generate(interpolator, (x_train,), obs)[1] for i=1:CHAINS]
traces = [[] for i=1:CHAINS]

#--------------
#Run Inference
#--------------
cd(current_dir)
println("Beginning Inference")
println("-------------------")

Threads.@threads for i=1:CHAINS
    #@inbounds println("Chain $i")
    #@inbounds println(starting_traces[i][:l])
    traces[i], _ = @inbounds RJNUTS(starting_traces[i], ITERS)
end

serialize(filename, traces)
