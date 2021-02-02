using RunTools
using MyModule

# Run from cmdline with: julia -L hyper.jl -E 'hyper(; params = Params(tags = [:leak]))' -- --queue
function hyper(; params = Params(), n = 10)
  params_ = merge(p(), params)
  paramsamples = rand(params_, n)
  display.(paramsamples)
  control(infer, paramsamples)
end