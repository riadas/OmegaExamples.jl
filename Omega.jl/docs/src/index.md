# Omega.jl

Omega.jl is a programming language for causal and probabilistic reasoning.
It was developed by [Anon Anon](http://Anon.org) with help from Anon Anon, Anon Anon, [Anon Anon](http://people.csail.mit.edu/xAnon/), [Anon Anon](https://cims.nyu.edu/~Anonr/) and [Anon Anon](https://people.csail.mit.edu/asolar/).

## Quick Start

Omega is built in Julia 1.0.  You can easily install it from a Julia repl with:

```julia
(v1.0) pkg> add Omega
```

Check Omega is working and gives reasonable results with: 

```julia
julia> using Omega

julia> rand(normal(0.0, 1.0))
0.7625637212030862
```

With that, see the [Tutorial](basictutorial.md) for a run through of the main features of Omega. 

## Contribute

We want your contributions!

- Probabilistic models
Please add probabilistic models and model families to https://github.com/Anon/OmegaModels.jl

- Inference procedures


## Citation

If you use Omega, please cite Omega papers:

[The Random Conditional Distribution for Uncertain Distributional Properties](http://www.Anon.org/publications/rcd.pdf)

```
@article{Anon2019rcd,
  title={The Random Conditional Distribution for Uncertain Distributional Properties},
  author={Anon, Anon and Anon, Anon and Anon, Anon and Anon, Anon and Anon, Anon Solar},
  journal={arXiv},
  year={2019}
}
```

[Soft Constraints for Inference with Declarative Knowedlge](http://www.Anon.org/publications/icmlsoft.pdf)

```
@article{Anon2019soft,
  title={Soft Constraints for Inference with Declarative Knowledge},
  author={Anon, Anon and Anon, Anon and Anon, Anon and Anon, Anon Solar and Anon, Anon},
  journal={arXiv preprint arXiv:1901.05437},
  year={2019}
}
```

If you use the causal inference features (`replace`), please cite:

[A Language for Counterfactual Generative Models](http://www.Anon.org/publications/causal.pdf)

```
@article{Anon2019counterfactual,
  title={Soft Constraints for Inference with Declarative Knowledge},
  author={Anon, Anon and Anon, Anon and Koppel, James and Anon, Anon Solar},
  year={2019}
}
```

## Acknowledgements

Omega leans heavily on the hard work of many packages and the Julia community as a whole, but in particular `Distributions.jl`, `Flux.jl`, and `Cassette.jl`.

## Index

```@contents
```

```@index
```
