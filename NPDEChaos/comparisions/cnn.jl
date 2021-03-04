begin
    using Plots
    using DiffEqFlux
    using OrdinaryDiffEq
    using Flux
    using Optim
    using JLD2
    using Zygote
    using SeqData
    using BlockArrays
    using LinearAlgebra
    using StaticArrays
    using BenchmarkTools
    using SparseArrays
    import LinearAlgebra.mul!

end

dt = 0.1
TRAIN = false
cluster = false
PLOT = false
EVAL = true

if cluster
    N_epochs = parse(Int, ARGS[3])
    SAVE_NAME = ARGS[2]
    gpu_type = parse(Int, ARGS[1])

    if length(ARGS) > 3
        N_t = parse(Int, ARGS[4])
    else
        N_t = 2500
    end

    if gpu_type == 1
        GPU = true
    else gpu_type == 2
        GPU = false
    end

    LOAD_DATA = false

else
    GPU = false
    SAVE_NAME = "train_cgle_onlycnn"
    N_epochs = 1000
    LOAD_DATA = false

    N_t = 25
end
COMPUTE_DATA = !(LOAD_DATA)

N_t_train = N_t
N_t_valid = 8000
N_t = N_t_train + N_t_valid + 1000
# do hyperparameter load / save up

n = 128
L = 192



if GPU
    using CuArrays
    using CuArrays.CUSPARSE
    LinearAlgebra.mul!(C::CuVector{T},adjA::Adjoint{<:Any,<:CuSparseMatrix},B::CuVector) where {T} = mv!('C',one(T),parent(adjA),B,zero(T),C,'O')
end

if !LOAD_DATA
    include("../scripts/cgle_fd.jl")
    lap = Δ

    @save string(SAVE_NAME,"-hyper.jld2") n L dt N_t t_start t_end N_epochs

    @save string(SAVE_NAME,"-data.jld2") dat_reim lap
    #old save @save string(SAVE_NAME,"-data.jld2") dat_reim n dx lap t_end

    @save string(SAVE_NAME,"-prob.jld2") prob_reim
else
    @load string(SAVE_NAME,"-hyper.jld2") n L dt N_t t_start t_end N_epochs

    include("../scripts/cgle_fd.jl")
    #old load @load string(SAVE_NAME,"-data.jld2") dat_reim n dx lap t_end
    @load string(SAVE_NAME,"-data.jld2") dat_reim lap
    @load string(SAVE_NAME,"-prob.jld2") prob_reim
    Δ = lap

end


println("--------")
println("Hyperparameter Overview")
println("n=", n, ", L=",L,", dt=",dt,", N_t=", N_t,", t_start=",t_start,", t_end=",t_end,", N_epochs=",N_epochs)
println("--------")

println("finished setting up/loading the data")
lap = nothing # just for garbage collector to make sure, the real laplacian is called Δ

const Ndim = n*n

if GPU
    CuArrays.allowscalar(false) # makes sure none of the slow fallbacks are used
    dat_reim = CuArray(Float32.(dat_reim))
else
    dat_reim = Float32.(dat_reim)
end

train, valid, test = SequentialData(dat_reim, 0, 1, N_t_train, N_t_valid, supervised=true);

#pars = gpu([cpu(α);cpu(Float32.(p_nn))]) # somehow this is needed due to vcat using scalar indices

function upsample(x)
  ratio = (2, 2, 1, 1)
  (h, w, c, n) = size(x)
  y = ones(eltype(x),(ratio[1], 1, ratio[2], 1, 1, 1))
  z = reshape(x, (1, h, 1, w, c, n))  .* y
  reshape(z, size(x) .* ratio)
end

function pad_uneven(x)
    (h, w, c, n) = size(x)
    y = similar(x, (h+2,w+2,c,1))
    fill!(y, 0)
    y[2:h+1,2:w+1,:,:] = x
    return y
end

m = Chain(x -> reshape(x,(n,n,2,1)), Conv((3,3),2=>8,swish,pad=(1,1)), MaxPool((2,2)), Conv((3,3),8=>8,swish,pad=(1,1)), MaxPool((2,2)), Conv((3,3),8=>2,swish, pad=(1,1)),Conv((3,3),2=>8,swish, pad=(1,1)), x -> upsample(x), Conv((3,3),8=>8,swish, pad=(1,1)), x -> upsample(x), Conv((3,3), 8=>2, pad=(1,1)), x -> reshape(x, (Ndim, 2)))


# define predict and loss functions
pars = Flux.params(m)

function predict(u0, N)
    z = similar(u0, (size(u0,1),size(u0,2),N))
    z[:,:,1] = u0
    for i=2:N
        z[:,:,i] = m(z[:,:,i-1])
    end
    z
end

loss_osa(x,y) = sum(abs2, m(x) .- y)

println("pre_compiling loss and predict functions")
loss_osa(train[1]...)


# next try the 1-step head thing
if TRAIN
    println("starting to train")
    opt = ADAMW()
    for i=1:N_epochs
        println("epoch ",i,"/",N_epochs)
        Flux.train!(loss_osa, pars, train, opt)
        if (i % 10) == 0
            #println("pars[1]=",pars[1],"...pars[2]=",pars[2])
            println("train m.e.: ",mean([loss_osa(train[i]...) for i=1:length(train)]))
            println("valid m.e.: ",mean([loss_osa(valid[i]...) for i=1:length(valid)]))
            @save string(SAVE_NAME,".jld2") m
        end
    end
    @save string(SAVE_NAME,".jld2") m

else
    @load string(SAVE_NAME,".jld2") m
    println(length(m))
    m = Chain(x -> reshape(x,(n,n,2,1)) ,m[1:6]..., x -> upsample(x), m[7], x -> upsample(x),m[8], x -> reshape(x, (Ndim, 2)))
    #@load string(SAVE_NAME,"-trained-prob.jld2") prob
    println("pars[1]=",pars[1],"...pars[2]=",pars[2])
    println("train (noise) m.e.: ",mean([loss_osa(train[i]...) for i=1:length(train)]))

    println("loaded model.")
end


PLOT = false
if EVAL
    N_prediction = 800
    prediction = predict(train[1][1], N_prediction+1)

    #prediction = reshape(array_to_complex(Array(prediction)), n, n, :)
    #truth = reshape(array_to_complex(dat_reim[:,:,1:N_prediction+1]), n, n, :)
    #diff_train = prediction - truth

    include("../scripts/eval_tools.jl")


    prediction_valid = predict(valid[1][1], N_prediction+1)

    prediction_valid = reshape(array_to_complex(Array(prediction_valid)), n, n, :)
    valid_index = train.N_t + 1
    truth_valid = reshape(array_to_complex(dat_reim[:,:,valid_index:valid_index+N_prediction]), n, n, :)
    diff_valid = prediction_valid - truth_valid

    δ = forecast_δ(abs.(prediction_valid), abs.(truth_valid), "norm")

    @save string("e-",SAVE_NAME,".jld2") δ_norm
    println(findall(δ_norm .> 0.4)[1])

    println("lyapunov times = ", findall(δ_norm .> 0.4)[1][2] * dt * 0.16724655)

    begin
        include("../scripts/eval_tools.jl")
        Plots.pyplot()
        δ1 = forecast_δ(prediction, truth)[1][:]
        plot(δ1, title=SAVE_NAME)
        savefig(string("plot-δ-",SAVE_NAME,".pdf"))

        δ2 = forecast_δ(prediction_valid, truth_valid)
        plot(δ2[1][:], title=SAVE_NAME)
        savefig(string("plot-δ-valid-",SAVE_NAME,".pdf"))

        @save string("Del-FC-",SAVE_NAME,".jld2") δ2

    end

    if PLOT
        Plots.pyplot()
        anim = Plots.@animate for i in 1:N_prediction+1
            Plots.heatmap(abs.(truth_valid[:,:,i]), clim=(-1.5,1.5))
        end

        gif(anim, string(SAVE_NAME, "-valid-truth-anim.gif"))

        anim3 = Plots.@animate for i in 1:N_prediction+1
            Plots.heatmap(abs.(prediction_valid[:,:,i]), clim=(-1.5,1.5))
        end

        gif(anim3, string(SAVE_NAME, "-valid-predict-anim.gif"))


        anim2 = Plots.@animate for i in 1:N_prediction+1
            Plots.heatmap(abs.(diff_valid[:,:,i]), clim=(-1.5,1.5))
        end

        gif(anim2, string(SAVE_NAME, "-valid-predict-diff-anim.gif"))


    end


end
