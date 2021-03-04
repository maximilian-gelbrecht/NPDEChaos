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
    using Distributions
    using CuArrays
    #using CUDA
end

begin
    using ColorSchemes
    cs_diverging = cgrad(:BrBG_11)
    cs_normal = cgrad(:GnBu_9)
end

dt = 0.1
TRAIN = false
cluster = false
PLOT = true
EVAL = true

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

    if length(ARGS) > 4
        long_integration = parse(Int, ARGS[5])
    else
        long_integration = 0
    end
    LONG_INTEGRATION = long_integration == 1 ? true : false

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
    SAVE_NAME = "train_ks_gpu-tau1-4096-Nt25-convtest-cuda90"
    #SAVE_NAME = "train_ks_gpu-tau1-4096-Nt50"
    #SAVE_NAME = "train_ks_gpu-tau1-4096-Nt25"
    N_epochs = 10
    LOAD_DATA = true
    LYAPUNOV = false
    N_t = 100
    STAB_NOISE_STD = 0.0
    N_WEIGHTS = 10
    LONG_INTEGRATION = false
    τ_max = 1
    RELOAD = false
end
COMPUTE_DATA = !(LOAD_DATA)

N_t_train = N_t
N_t_valid = 8000
N_t = N_t_train + N_t_valid + 1000
# do hyperparameter load / save up


n = 4096
#n = 2048
#L = 580
L = 1160
#n = 8192
#L = 2320


if GPU
    using CuArrays
    using CuArrays.CUSPARSE
    LinearAlgebra.mul!(C::CuVector{T},adjA::Adjoint{<:Any,<:CuSparseMatrix},B::CuVector) where {T} = mv!('C',one(T),parent(adjA),B,zero(T),C,'O')
    #using CUDA
    #using CUDA.CUSPARSE
end

if !LOAD_DATA
    include("scripts/ks_fd.jl")

    @save string(SAVE_NAME,"-hyper.jld2") n L dt N_t N_t_train N_t_valid t_start t_end N_epochs LONG_INTEGRATION STAB_NOISE_STD N_WEIGHTS τ_max

    @save string(SAVE_NAME,"-data.jld2") dat
    #old save @save string(SAVE_NAME,"-data.jld2") dat_reim n dx lap t_end

    @save string(SAVE_NAME,"-prob.jld2") prob
else
    @load string(SAVE_NAME,"-hyper.jld2") n L dt N_t N_t_train N_t_valid t_start t_end N_epochs LONG_INTEGRATION STAB_NOISE_STD N_WEIGHTS τ_max

    include("scripts/ks_fd.jl")
    @load string(SAVE_NAME,"-data.jld2") dat
    @load string(SAVE_NAME,"-prob.jld2") prob

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

const Ndim = n

function fd_derivative(n,dx::T) where T<:Number
  ∂x = (diagm(1=>ones(n-1)) + diagm(-1=>-1*ones(n-1)))
  ∂x[1,end] = -1
  ∂x[end,1] = 1
  ∂x ./= (2*dx)
  sparse(T.(∂x))
end

if GPU
    P = CuArray(ones(Float32,1,Ndim))
    ∂x = CuArrays.CUSPARSE.CuSparseMatrixCSC(fd_derivative(n, Float32(dx)))
    ∂x2 = CuArrays.CUSPARSE.CuSparseMatrixCSC(Float32.(∂x2))
    m∂x4 = CuArrays.CUSPARSE.CuSparseMatrixCSC(Float32.(-∂x4))


    #∂x = CUSPARSE.switch2bsr(∂x,convert(Cint,16))
    #∂x2 = CUSPARSE.switch2bsr(∂x2,convert(Cint,16))
    #m∂x4 = CUSPARSE.switch2bsr(m∂x4,convert(Cint,16))

else
    P = ones(Float32,1,Ndim)
    ∂x = fd_derivative(n, Float32(dx))
    ∂x2 = Float32.(∂x2)
    m∂x4 = Float32.(-∂x4)
end

include("scripts/nn_tools.jl")




#nn = Chain(NablaSkipConnection2(Float32(rand(Uniform(0.3, 0.5)))), NablaSkipConnection2(Float32(rand(Uniform(0.3, 0.5)))), NablaSkipConnection2(Float32(rand(Uniform(0.5, 0.7)))), NablaSkipConnection2(Float32(rand(Uniform(0.5, 0.7)))), x->transpose(x), DenseGPU(1, n_weights, swish), DenseGPU(n_weights, n_weights, swish), DenseGPU(n_weights, 1)) |> gpu

nn = Chain(NablaSkipConnection2(Float32(rand(Uniform(0.3, 0.5))),GPU), NablaSkipConnection2(Float32(rand(Uniform(0.3, 0.5))),GPU), NablaSkipConnection2(Float32(rand(Uniform(0.5, 0.7))),GPU), NablaSkipConnection2(Float32(rand(Uniform(0.5, 0.7))),GPU), x->transpose(x), SkipConnection(Chain(DenseGPU(1, n_weights, swish), DenseGPU(n_weights, n_weights, swish), DenseGPU(n_weights, 1)),+)) |> gpu
println("Nabla Pars")
println(Flux.params(nn[1:4]))
println("----")
p_nn, re_nn = Flux.destructure(nn)
println(p_nn)
println(p_nn[1])
if STAB_NOISE_STD > 0.0
    stabilization_noise = Normal(0.,STAB_NOISE_STD)
else
    stabilization_noise = nothing
end



if GPU
    #CuArrays.allowscalar(false)
    CUDA.allowscalar(false)
    dat = CuArray(Float32.(dat))
else
    dat = Float32.(dat)
end

train, valid, test = SequentialData(dat, 0, 1, N_t_train, N_t_valid, supervised=true, stabilization_noise = stabilization_noise);

train_no_noise, valid_no_noise, test_no_noise = SequentialData(dat, 0, 1, N_t_train, N_t_valid, supervised=true, stabilization_noise = stabilization_noise);
println("---")
println("length of train set = ",length(train))
println("---")
println("length of valid set = ",length(valid))
println("---")



function ks_oop(u,p,t)
        m∂x4*u - reshape(transpose(re_nn(p)(u)),:) - u.*(∂x*u)
end


#pars = gpu([cpu(α);cpu(Float32.p_nn))]) # somehow this is needed due to vcat using scalar indices
pars = gpu(p_nn)
prob = ODEProblem(ks_oop, train[1][1], (Float32(0.),Float32(dt)), pars)

if GPU
    predict_osa(u0, p) = CuArray(concrete_solve(prob, Tsit5(), u0, p,saveat = [Float32(dt)],
    reltol=Float32(1e-3), abstol=Float32(1e-4)))[:,1]
    predict_longer(u0, p) = CuArray(concrete_solve(remake(prob, tspan=(0f0, Float32(15*dt))), Tsit5(), u0, p,saveat = [Float32(dt)],
    reltol=Float32(1e-3), abstol=Float32(1e-4)))[:,1]
    predict(u0, p) = CuArray(concrete_solve(remake(prob, tspan=(0f0, Float32(t_end - t_start))), Tsit5(), u0, p,saveat = Float32(dt)))
    predict(u0, p, N) = CuArray(concrete_solve(remake(prob, tspan=(0f0, Float32(dt)*N)), Tsit5(), u0, p,saveat = Float32(dt)))
    predict_op_only(u0, p, N) = CuArray(concrete_solve(remake(prob_op_only, tspan=(0f0, Float32(dt)*N)), Tsit5(), u0, p,saveat = Float32(dt)))
    predict_truth(u0, N) = CuArray(solve(ODEProblem(cgle_fd_reim!,u0,(0f0, Float32(dt)*N),[α; imag(β)]),Tsit5(),saveat=Float32(dt))) # do this newu0=u0, tspan=(0f0, Float32(dt)*N)),Tsit5(), saveat=Float32(dt)))

else
    predict_osa(u0, p) = Array(concrete_solve(prob, Tsit5(), u0, p,saveat = [Float32(dt)],
    reltol=Float32(1e-3), abstol=Float32(1e-4)))[:,1]
    predict_longer(u0, p) = Array(concrete_solve(remake(prob, tspan=(0f0, Float32(15*dt))), Tsit5(), u0, p,saveat = [Float32(dt)],
        reltol=Float32(1e-3), abstol=Float32(1e-4)))[:,1]
    predict(u0, p) = concrete_solve(remake(prob, tspan=(0f0, Float32(t_end - t_start))), Tsit5(), u0, p,saveat = Float32(dt))
    predict(u0, p, N) = concrete_solve(remake(prob, tspan=(0f0, Float32(dt)*N)), Tsit5(), u0, p,saveat = Float32(dt))
    predict_op_only(u0, p, N) = concrete_solve(remake(prob_op_only, tspan=(0f0, Float32(dt)*N)), Tsit5(), u0, p,saveat = Float32(dt))
    predict_truth(u0, N) = Array(solve(ODEProblem(cgle_fd_reim!,u0,(0f0, Float32(dt)*N),[α; imag(β)]),Tsit5(),saveat=Float32(dt)))
end

#nabla_penalty_func(x) = 4f0*abs((2f0*((x-0.5f0)))^6 - 1f0)
nabla_penalty_func_vec(x) =  4f0 .* abs.((2f0.*((x .- 0.5f0))).^6 .- 1f0)

loss_osa(p, x, y) = sum(abs2, predict_osa(x, p) - y) + sum(nabla_penalty_func_vec(p[1:4])) + 1e-5*sum(abs.(p[5:end]))

loss_no_penalty(p, x, y) = sum(abs2, predict_osa(x, p) - y)
 # there predict_longer here before
println("pre_compiling loss and predict functions")
predict_osa(train[1][1], pars)
loss_osa(pars, train[1]...)


# next try the 1-step head thing
n_valid_err = 0
valid_err = 1e20

if TRAIN
    println("starting to train")

    for i_τ = 1:τ_max
        println("initiliazing tau=", i_τ)
        global pars,valid_err, n_valid_err

        train_long, valid_long, test_long =
            SequentialData(dat, 0, i_τ, N_t_train, N_t_valid, supervised = true, stabilization_noise = stabilization_noise)

        prob_long = remake(prob, tspan = (0.0f0, Float32(i_τ*dt)))

        """
        predict_long(u0, p) = CuArray(concrete_solve(
                prob_long,
                Tsit5(),
                u0,
                p,
                saveat = Float32.(dt:dt:i_τ*dt),
                reltol = Float32(1e-6),
            ))
        """

        predict_long(u0, p) = Array(concrete_solve(
                    prob_long,
                    Tsit5(),
                    u0,
                    p,
                    saveat = Float32.(dt:dt:i_τ*dt),
                    reltol = Float32(1e-6),
                ))

        println("precompiling ...")
        #println(predict_long(train_long[1][1][:,:,1], pars))

        loss_long(p, x, y) = sum(abs2, predict_long(x[:, 1], p) - y) + sum(nabla_penalty_func_vec(p[1:4])) + 1e-4*sum(abs,p[5:end])

        loss_long_no_penalty(p, x, y) = sum(abs2, predict_long(x[:, 1], p) - y)

        loss_long(pars, train_long[1]...)
        loss_long_no_penalty(pars, train_long[1]...)

        println("...done")

        N_e = N_epochs
        if i_τ == 1
            N_e *= 10
        end

        for i_all=1:N_e
            println("tau= ",i_τ," , epoch: ", i_all, "/", N_epochs)


            global res
            res = DiffEqFlux.sciml_train(loss_long, pars, ADAMW(), train_long)# look into diffeqflux
            pars = res.minimizer
            #println(loss_adjoint(res.minimizer))

            println(pars[1:4])
            if (i_all % 10) == 0
                begin
                    #new_valid_err = mean([loss_long_no_penalty(pars, valid[i]...) for i=1:500])

                    #if new_valid_err > valid_err
                    #    n_valid_err += 1
                    #else
                    #    n_valid_err = 0
                    #end

                    #valid_err = new_valid_err

                    #println("valid (no noise) m.e.: ", valid_err)

                    pars = cpu(pars)
                    @save string(SAVE_NAME,"-pars.jld2") pars
                    pars = gpu(pars)

                    #println("train (noise) m.e.: ",mean([loss_long_no_penalty(pars, train[i]...) for i=1:length(train)]))

		            #println("train (penalty) m.e.: ",mean([loss_long(pars, train_no_noise[i]...) for i=1:length(train_no_noise)]))
                    #println("valid (no noise) m.e.: ", mean([loss_long_no_penalty(pars, valid_no_noise[i]...) for i=1:100]))
                end
            end

            #if n_valid_err > 2
            #    println("valid error does not decrease, ending training...")
            #    break
            #end

        end
        pars = cpu(pars)
        @save string(SAVE_NAME, "-pars.jld2") pars
        pars = gpu(pars)
    end
else
    @load string(SAVE_NAME,"-pars.jld2") pars
    #@load string(SAVE_NAME,"-trained-prob.jld2") prob
    println("pars[1]=",pars[1],"...pars[2]=",pars[2])
    println("train (noise) m.e.: ",mean([loss_no_penalty(pars, train[i]...) for i=1:length(train)]))

    println("loaded model.")
end

PLOT = false
if EVAL
    include("scripts/eval_tools.jl")
    N_prediction = 800

    if N_prediction > valid.N_t
        N_prediction = valid.N_t
    end
    #prediction = predict(train[1][1], pars, N_prediction)

    #prediction = reshape(array_to_complex(Array(prediction)), 50, 50, :)
    #truth = reshape(array_to_complex(dat_reim[:,:,1:N_prediction+1]), 50, 50, :)
    #diff_train = prediction - truth
    println("computing prediction")
    prediction_valid = predict(valid[1][1], pars, N_prediction)
    prediction_valid = Array(prediction_valid)
    valid_index = train.N_t + 1
    truth_valid = dat[:,valid_index:valid_index+N_prediction]
    diff_valid = prediction_valid - truth_valid


    begin
        Plots.pyplot()
        #δ1 = forecast_δ(prediction, truth)[1][:]
        #plot(δ1, title=SAVE_NAME)
        #savefig(string("plot-δ-",SAVE_NAME,".pdf"))

        δ2 = forecast_δ(prediction_valid, truth_valid)
        plot(δ2[1][:], title=SAVE_NAME)
        savefig(string("plot-δ-valid-",SAVE_NAME,".pdf"))

        @save string("Del-FC-",SAVE_NAME,".jld2") δ2

    end

    if PLOT
        Plots.pyplot()

        Plots.heatmap(diff_valid)



    end


end

#Plots.heatmap(diff_valid[1:256,1:600], size=(500,300))
# investrigate if nn is identity
#begin
#    x = range(-2.,2.,length=Ndim)
#    idd = (re_nn(pars)[6:end](transpose(x)))[:] - x
#    plot(x,idd)
#end
PAPER_PLOT=true
if PAPER_PLOT
    ticks=[200,400]

    p1 = Plots.heatmap(prediction_valid, c=cs_diverging,size=(750,375),dpi=300)
    p3 = Plots.heatmap(diff_valid, c=cs_diverging,size=(750,375),dpi=300)
    p2 = Plots.heatmap(prediction_valid[1:256,1:600], c=cs_diverging,size=(375,375),xticks=ticks,dpi=300)
    p4 = Plots.heatmap(diff_valid[1:256,1:600], c=cs_diverging,size=(375,375),xticks=ticks,dpi=300)


    Plots.plot(p1,p2,p3,p4, layout=grid(2,2, widths=[0.65,0.35,0.65,0.35]))
    Plots.savefig(string("plot-paper-ks-",SAVE_NAME,".png"))

    δ = forecast_δ(prediction_valid, truth_valid, "norm")
    #println(findall(δ .> 0.4)[1])

    #@save string("e-",SAVE_NAME,".jld2") δ

end

begin
    import PyPlot
    λmax = 0.07
    dt = 0.1

    XLIMS = [0, 800]

    fig = PyPlot.plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(0,1)
    ax1.set_xlabel("Integration time steps")
    ax1.set_xlim(XLIMS[1],XLIMS[2])
    PyPlot.plt.yscale("log")
    ax1.set_ylim(0.,1.5)
    ax2.set_xlim(XLIMS[1],XLIMS[2]*λmax*dt)
    PyPlot.savefig("axenks.pdf")
end
