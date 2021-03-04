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
    using CuArrays

    using ColorSchemes
end

cs_diverging = cgrad(:BrBG_11)
cs_normal = cgrad(:GnBu_9)

dt = 0.1
TRAIN = false
cluster = false
PLOT = false
EVAL = true

"""
if cluster==true this script is meant to be called from the command line with extra arguments (e.g. by a batch system such as SLURM)

the extra arguments are
* 1) GPU, 1==true, 2==false
* 2) SAVE_NAME, base name of the saved files
* 3) N_EPOCHS, number of the training epochs (for N_t=1 the 10-fold value is used)
* 4) LYAPUNOV, alternate mode for computing the Lyapunov exponents, does not train the NPDE
* 5) legacy, not used anymore
* 6) N_t, length of training dataset
* 7) RELOAD, reload data

"""



# input arguments when called as a script with a cluster manager like slurm
if cluster
    N_epochs = parse(Int, ARGS[3])
    SAVE_NAME = ARGS[2]
    gpu_type = parse(Int, ARGS[1])

    if length(ARGS) > 3
        LYAPUNOV = parse(Int, ARGS[4])
    else
        LYAPUNOV = 0
    end
    LYAPUNOV = (LYAPUNOV == 1) ? true : false

    if length(ARGS) > 5
        N_t = parse(Int, ARGS[6])
    else
        N_t = 2500
    end

    if length(ARGS) > 6
        τ_max = parse(Int, ARGS[7])
    else
        τ_max = 15
    end

    if length(ARGS) > 7
        RELOAD = parse(Int, ARGS[8])
    else
        RELOAD = 0
    end
    RELOAD = RELOAD == 1 ? true : false


    if gpu_type == 1
        GPU = true
    else gpu_type == 2
        GPU = false
    end

    LOAD_DATA = false

    if LYAPUNOV
        TRAIN = false
        LOAD_DATA = true
    end
    STAB_NOISE_STD = 0.0
    N_WEIGHTS = 10
    if RELOAD
        LOAD_DATA = true
    end
else
    GPU = false
    #SAVE_NAME = "train_cgle_128-2d-gpu-tau1-Nt50-Ne20-reg"
    #SAVE_NAME = "train_cgle_128-2d-gpu-tau1-Nt25-Ne20-convtest"
    SAVE_NAME = "train_cgle_128-2d-gpu-tau1-Nt25-Ne20-convtest"
    N_epochs = 100
    LOAD_DATA = true
    LYAPUNOV = false
    N_t = 20000
    STAB_NOISE_STD = 0.0
    N_WEIGHTS = 10
    LONG_INTEGRATION = false
    τ_max = 15
    RELOAD = false

end
COMPUTE_DATA = !(LOAD_DATA)

N_t_train = N_t
N_t_valid = 8000
N_t = N_t_train + N_t_valid + 1000

# system parameters
n = 128
L = 192

# there is/was a bug within CuArrays that is fixed here provisionally
if GPU
    using CuArrays
    using CuArrays.CUSPARSE
    LinearAlgebra.mul!(C::CuVector{T},adjA::Adjoint{<:Any,<:CuSparseMatrix},B::CuVector) where {T} = mv!('C',one(T),parent(adjA),B,zero(T),C,'O')
end

if !LOAD_DATA
    include("scripts/cgle_fd.jl")
    lap = Δ

    @save string(SAVE_NAME,"-hyper.jld2") n L dt N_t N_t_train N_t_valid t_start t_end N_epochs LONG_INTEGRATION STAB_NOISE_STD N_WEIGHTS τ_max

    @save string(SAVE_NAME,"-data.jld2") dat_reim lap
    @save string(SAVE_NAME,"-prob.jld2") prob_reim
else
    @load string(SAVE_NAME,"-hyper.jld2") n L dt N_t N_t_train N_t_valid t_start t_end N_epochs LONG_INTEGRATION STAB_NOISE_STD N_WEIGHTS τ_max

    include("scripts/cgle_fd.jl")
    @load string(SAVE_NAME,"-data.jld2") dat_reim lap
    @load string(SAVE_NAME,"-prob.jld2") prob_reim
    Δ = lap

    if RELOAD
        @load string(SAVE_NAME,"-pars.jld2") pars
        pars = gpu(pars)
        println("Loaded parameters from old training run.")
    end
end
const n_weights = N_WEIGHTS

println("--------")
println("Hyperparameter Overview")
println("n=", n, ", L=",L,", dt=",dt,", N_t=", N_t,", N_t_train=",N_t_train,", N_t_valid=",N_t_valid," , t_start=",t_start,", t_end=",t_end,", N_epochs=",N_epochs,", LONG_INTEGRATION=",LONG_INTEGRATION,", STAB_NOISE_STD=",STAB_NOISE_STD,", N_WEIGHTS=",N_WEIGHTS, " tau_max=",τ_max)
println("--------")

println("finished setting up/loading the data")
lap = nothing # just for garbage collector to make sure, the real laplacian is called Δ

const Ndim = n*n

# projection matrix used by the GPU fitted version of Dense
if GPU
    P = CuArray(ones(Float32,1,Ndim))
else
    P = ones(Float32,1,Ndim)
end

include("scripts/nn_tools.jl")

# define the neural net
nn = Chain(DenseGPU(2, n_weights, swish), DenseGPU(n_weights, n_weights, swish), DenseGPU(n_weights, n_weights, swish), DenseGPU(n_weights, 2)) |> gpu

#nn = Chain(DenseGPU(2, n_weights, swish), DenseGPU(n_weights, 2*n_weights, swish), DenseGPU(2*n_weights, n_weights, swish), DenseGPU(n_weights, 2)) |> gpu

#nn = Chain(DenseGPU(2, n_weights, swish), DenseGPU(n_weights, n_weights, swish), DenseGPU(n_weights, 2)) |> gpu

p_nn, re_nn = Flux.destructure(nn)

if STAB_NOISE_STD > 0.0
    stabilization_noise = Normal(0.,STAB_NOISE_STD)
else
    stabilization_noise = nothing
end

if GPU
    Δ = CuArrays.CUSPARSE.CuSparseMatrixCSC(Float32.(-Δ));
    α = CuArray(Float32[2.0f0])
    CuArrays.allowscalar(false) # makes sure none of the slow fallbacks are used
    const matRe = reshape(CuArray(Float32[1,0]),(1,2))
    const matIm = reshape(CuArray(Float32[0,1]),(1,2))
    dat_reim = CuArray(Float32.(dat_reim))
else
    Δ = Float32.(-Δ); # sign?
    α = Float32[2.0f0]
    const matRe = reshape(Float32[1,0],(1,2))
    const matIm = reshape(Float32[0,1],(1,2))
    dat_reim = Float32.(dat_reim)
end

train, valid, test = SequentialData(dat_reim, 0, 1, N_t_train, N_t_valid, supervised=true, stabilization_noise = stabilization_noise);

println("---")
println("length of train set = ",length(train))
println("---")
println("length of valid set = ",length(valid))
println("---")

function ode_reim_oop(u, p, t)
    ReU = @view u[:,1]
    ImU = @view u[:,2]
    nn_res = re_nn(p)(transpose(u))
    return (Δ*(ReU - α.*ImU))*matRe + (Δ*(ImU + α.*ReU))*matIm + transpose(nn_res)
end


#pars = gpu([cpu(α);cpu(Float32.(p_nn))]) # somehow this is needed due to vcat using scalar indices

pars = gpu(p_nn)
prob = ODEProblem(ode_reim_oop, train[1][1], (Float32(0.),Float32(dt)), pars)

# This is for a comparision: how do the prediction look if we actually only take the "known" part of the model
begin
     function ode_operator_only(u,p,t)
         ReU = @view u[:,1]
         ImU = @view u[:,2]
         #nn_res = nn(transpose(u), p)
         return (Δ*(ReU - α.*ImU))*matRe + (Δ*(ImU + α.*ReU))*matIm
     end
     prob_op_only = ODEProblem(ode_operator_only, train[1][1], (Float32(0.),Float32(dt)), pars)
 end

# a lot of different predict functions for different situations.
if GPU
    predict_osa(u0, p) = CuArray(concrete_solve(prob, Tsit5(), u0, p,saveat = [Float32(dt)],
    reltol=Float32(1e-3), abstol=Float32(1e-4)))[:,:,1]
    predict_longer(u0, p) = CuArray(concrete_solve(remake(prob, tspan=(0f0, Float32(15*dt))), Tsit5(), u0, p,saveat = [Float32(dt)],
    reltol=Float32(1e-3), abstol=Float32(1e-4)))[:,:,1]
    predict(u0, p) = CuArray(concrete_solve(remake(prob, tspan=(0f0, Float32(t_end - t_start))), Tsit5(), u0, p,saveat = Float32(dt)))
    predict(u0, p, N) = CuArray(concrete_solve(remake(prob, tspan=(0f0, Float32(dt)*N)), Tsit5(), u0, p,saveat = Float32(dt)))
    predict_op_only(u0, p, N) = CuArray(concrete_solve(remake(prob_op_only, tspan=(0f0, Float32(dt)*N)), Tsit5(), u0, p,saveat = Float32(dt)))
    predict_truth(u0, N) = CuArray(solve(ODEProblem(cgle_fd_reim!,u0,(0f0, Float32(dt)*N),[α; imag(β)]),Tsit5(),saveat=Float32(dt)))
else
    predict_osa(u0, p) = Array(concrete_solve(prob, Tsit5(), u0, p,saveat = [Float32(dt)],
    reltol=Float32(1e-3), abstol=Float32(1e-4)))[:,:,1]
    predict_longer(u0, p) = Array(concrete_solve(remake(prob, tspan=(0f0, Float32(15*dt))), Tsit5(), u0, p,saveat = [Float32(dt)],
        reltol=Float32(1e-3), abstol=Float32(1e-4)))[:,:,1]
    predict(u0, p) = concrete_solve(remake(prob, tspan=(0f0, Float32(t_end - t_start))), Tsit5(), u0, p,saveat = Float32(dt))
    predict(u0, p, N) = concrete_solve(remake(prob, tspan=(0f0, Float32(dt)*N)), Tsit5(), u0, p,saveat = Float32(dt))
    predict_op_only(u0, p, N) = concrete_solve(remake(prob_op_only, tspan=(0f0, Float32(dt)*N)), Tsit5(), u0, p,saveat = Float32(dt))
    predict_truth(u0, N) = Array(solve(ODEProblem(cgle_fd_reim!,u0,(0f0, Float32(dt)*N),[α; imag(β)]),Tsit5(),saveat=Float32(dt)))
end


loss_osa(p, x, y) = sum(abs2, predict_osa(x, p) .- y) # there predict_longer here before
println("pre_compiling loss and predict functions")
predict_osa(train[1][1], pars)
loss_osa(pars, train[1]...)

train_longer, valid_longer, test_longer = SequentialData(dat_reim, 0, 15, N_t_train, N_t_valid, supervised=true);

loss_longer(p, x, y) = sum(abs2, predict_longer(x[:,:,1], p) .- y)
println("...")
predict_longer(train_longer[1][1][:,:,1], pars)
loss_longer(pars, train_longer[1]...)
println("...finished")

n_valid_err = 0
valid_err = 1e20

if TRAIN
    println("starting to train")

    for i_τ = 1:τ_max
        println("initiliazing tau=", i_τ)
        global pars, n_valid_err, valid_err

        train_long, valid_long, test_long =
            SequentialData(dat_reim, 0, i_τ, N_t_train, N_t_valid, supervised = true, stabilization_noise = stabilization_noise)

        prob_long = remake(prob, tspan = (0.0f0, Float32(i_τ*dt)))

        predict_long(u0, p) = CuArray(concrete_solve(
                prob_long,
                Tsit5(),
                u0,
                p,
                saveat = Float32.(dt:dt:i_τ*dt),
                reltol = Float32(1e-6),
            ))

        println("precompiling ...")

        loss_long(p, x, y) = sum(abs2, predict_long(x[:, :, 1], p) .- y)
        loss_long(pars, train_long[1]...)
        println("...done")

        # OSA gets many more epochs of training because it is consideriably faster
        N_e = N_epochs
        if i_τ == 1
            N_e *= 10
        end

        for i_all=1:N_e
            println("tau= ",i_τ," , epoch: ", i_all, "/", N_epochs)

            global res
            res = DiffEqFlux.sciml_train(loss_long, pars, ADAMW(), train_long)
            pars = res.minimizer


            if (i_all % 10) == 0

                new_valid_err = mean([loss_osa(pars, valid[i]...) for i=1:200])

                if new_valid_err > valid_err
                    n_valid_err += 1
                else
                    n_valid_err = 0
                end

                valid_err = new_valid_err

                begin
                    # there's a weird error with CuSparse sometimes when printing / computing these errors, not sure why
                    #println("train (noise) m.e.: ",mean([loss_osa(pars, train[i]...) for i=1:length(train)]))
                    println("valid (no noise) m.e.: ", valid_err)

                    pars = cpu(pars)
                    @save string(SAVE_NAME,"-pars.jld2") pars
                    pars = gpu(pars)


                end
            end

            if n_valid_err > 2
                println("valid error does not decrease, ending training...")
                break
            end

        end
        pars = cpu(pars)
        @save string(SAVE_NAME, "-pars.jld2") pars
        pars = gpu(pars)
    end
else
    @load string(SAVE_NAME,"-pars.jld2") pars
    println("pars[1]=",pars[1],"...pars[2]=",pars[2])
    println("train (noise) m.e.: ",mean([loss_osa(pars, train[i]...) for i=1:length(train)]))

    println("loaded model.")
end

PLOT = false
if EVAL
    N_prediction = 700

    if N_prediction > valid.N_t
        N_prediction = valid.N_t
    end
    prediction = predict(train[1][1], pars, N_prediction)



    prediction_valid = predict(valid[1][1], pars, N_prediction)

    prediction_valid = reshape(array_to_complex(Array(prediction_valid)), n,  n, :)
    valid_index = train.N_t + 1
    truth_valid = reshape(array_to_complex(dat_reim[:,:,valid_index:valid_index+N_prediction]), n, n, :)
    diff_valid = prediction_valid - truth_valid

    include("scripts/eval_tools.jl")

    begin
        Plots.pyplot()
        println("lyapunov times = ", findall(δ_norm .> 0.4)[1][2] * dt * 0.16724655)

        println(findall(δ_norm .> 0.3)[1])

        println("lyapunov times = ", findall(δ_norm .> 0.3)[1][2] * dt * 0.16724655)
    end

    # animations
    if PLOT
        Plots.pyplot()
        anim = Plots.@animate for i in 1:N_prediction+1
            Plots.heatmap(abs.(truth_valid[:,:,i]), clim=(-1.5,1.5), title=string("Step ",i))
        end

        gif(anim, string(SAVE_NAME, "-valid-truth-anim.gif"))

        anim3 = Plots.@animate for i in 1:N_prediction+1
            Plots.heatmap(abs.(prediction_valid[:,:,i]), clim=(-1.5,1.5), title=string("Step ",i))
        end

        gif(anim3, string(SAVE_NAME, "-valid-predict-anim.gif"))


        anim2 = Plots.@animate for i in 1:N_prediction+1
            Plots.heatmap(abs.(diff_valid[:,:,i]), clim=(-1.5,1.5), title=string("Step ",i))
        end

        gif(anim2, string(SAVE_NAME, "-valid-predict-diff-anim.gif"))
    end
end

# main plot for the paper
PAPER_PLOT=true
if PAPER_PLOT
    times=[100, 150, 200,  250]
    ticks=[0,32,64,96,128]
    plots_fc = []
    for itimes in times

        push!(plots_fc, Plots.heatmap(abs.(prediction_valid[:,:,itimes]),clim=(0.,1.5),c=cs_normal, aspect_ratio=1, xlims=[0,n], ylims=[0,n], size=(400,400), xticks=ticks,yticks=ticks))
        push!(plots_fc, Plots.heatmap(abs.(diff_valid[:,:,itimes]),clim=(0.,1.5),c=cs_diverging, aspect_ratio=1,xlims=[0,n], ylims=[0,n], size=(400,400),xticks=ticks,yticks=ticks))

    end
    Plots.plot(plots_fc..., layout=grid(length(times),2))
    Plots.savefig(string("plot-paper-cgle-",SAVE_NAME,".pdf"))
end

SAVE_FORECAST_ERROR=true
if SAVE_FORECAST_ERROR==true
    δ = forecast_δ(abs.(prediction_valid), abs.(truth_valid), "norm")
    @save string("e-",SAVE_NAME,".jld2") δ
end

# this block compares operator only and the actual full model
COMPARE_OP_ONLY = false
if COMPARE_OP_ONLY
    include("scripts/eval_tools.jl")
    N_prediction = 700
    prediction = predict_op_only(valid[1][1], pars, N_prediction)
    prediction = reshape(array_to_complex(Array(prediction)), 128, 128, :)

    truth = reshape(array_to_complex(dat_reim[:,:,valid_index:valid_index+N_prediction]), 128, 128, :)

    diff_op = prediction - truth

    δ = forecast_δ(abs.(prediction), abs.(truth), "norm")
    @save string("e-op-only",SAVE_NAME,".jld2") δ

    if PLOT

        plot(δ2[1][:], title=SAVE_NAME)
        savefig(string("plot-δ-op-only",SAVE_NAME,".pdf"))

        Plots.pyplot()
        anim = Plots.@animate for i in 1:N_prediction+1
            Plots.heatmap(abs.(diff_op[:,:,i]), clim=(-1.5,1.5))
        end

        gif(anim, string(SAVE_NAME, "-op-only-diff-anim.gif"))
    end
end
