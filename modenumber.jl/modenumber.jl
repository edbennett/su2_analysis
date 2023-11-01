using Pkg
# Pkg.add(url="https://gitlab.com/pietro-butti/jkvar.jl.git")
# Pkg.add(url="https://gitlab.com/pietro-butti/modenumber.jl.git")

println("CONTROL: loading packages...")
using DelimitedFiles, Printf, PyPlot, Statistics, jkvar, modenumber, Measurements, LsqFit, LaTeXStrings, ProgressMeter, ArgParse
println("CONTROL: loading packages...done!")

##

s = ArgParseSettings()

@add_arg_table! s begin
    "--label"
        help = "Name of the ensemble"
    "--beta"
        help = "Lattice coupling of the ensemble"
        arg_type = Float64
    "--mass"
        help = "Fermion mass of the ensemble"
        arg_type = Float64
    "--volume"
        help = "Number of sites"
        arg_type = Int
    "--Nconf_boot"
        help = "Number of bootstrap samples"
        arg_type = Int
        default = 1000
    "--Npick"
        help = "Number of configurations"
        arg_type = Int
    "--omega_min_lower"
        help = "Minimum value of lower end of window"
        arg_type = Float64
    "--omega_min_step"
        help = "Step size of scan in lower end of window"
        arg_type = Float64
    "--omega_min_upper"
        help = "Maximum value of lower end of window"
        arg_type = Float64
    "--delta_omega_lower"
        help = "Minimum value of size of window"
        arg_type = Float64
    "--delta_omega_step"
        help = "Step size of scan in window size"
        arg_type = Float64
    "--delta_omega_upper"
        help = "Maximum value of size of window"
        arg_type = Float64
    "--output"
        help = "Output file"
    "--input_path"
        help = "Path to input file"
        default = "."
    "--input_filename"
        help = "Input filename"
        default = "."
    "--input_type"
        help = "Type of input file (HiRep or xyerrfile)"
        default = "HiRep"
end

args = parse_args(s)

e = ensemble(args["label"], args["beta"], args["mass"], args["volume"],
             path = args["input_path"], name = args["input_filename"],
             input_type = args["input_type"])

Nconf_boot = args["Nconf_boot"]
Npick      = args["Npick"]
omega_min  = args["omega_min_lower"]:args["omega_min_step"]:args["omega_min_upper"]
delta_omega = args["delta_omega_lower"]:args["delta_omega_step"]:args["delta_omega_upper"]

saveto     = args["output"]


X = e.M
Y = e.nu
aux = sort([(xx,yy) for (xx,yy) in zip(X,Y)], by = x -> x[1])
X = [a[1] for a in aux]
Y = [a[2] for a in aux]

f,fails,ranges = windowing(X,Y,omega_min,delta_omega,Nconf_boot,Npick)
gfin,errg,w,g = weight_results(f,fails,ranges,Nconf_boot)

f = open(saveto,"w")
println(f,"# Results for ensemble $(e.tag)")
println(f,"# γ* = $gfin     (syst. err. = $errg)")
println(f,"xmin    xmax         gamma       err(stat)        weight")
for i in 1:length(w)
    @printf(f,
    "%.4f  %.4f       %.8f  %.8f       %.8f\n",
        ranges[i][1],
        ranges[i][2],
        val(g[i]),
        err(g[i]),
        w[i]
    )
end
close(f)

