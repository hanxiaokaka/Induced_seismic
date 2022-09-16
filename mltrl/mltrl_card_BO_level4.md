# MLTRL Card example - Seismic Analysis for Induced Forecasts  - level 8
| Summary info        | Content, links       |
| -------------------------- | ------------- |
| Tech name (project ID)  | Seismic Analysis for Induced Forecasts   |
| Current Level           | 4 (*link to prior level cards*) |
| Owner(s)                | Giuseppe Castiglione, Alexandre Chen, Akshay Suresh, Han Xiao, Kayla Kroll, Christopher Sherman, Constantin Weisser|
| Reviewer(s)             |                           |
| Main project page       | (*link to e.g. team's internal wiki page*)   |
| Req's, V&V docs         | (*link, if not on main page above*)   |
| Code repo & docs        | (*link to Github etc*)   |
| Ethics checks?          | WIP: *link to MVBO [checklist](../ethics_checklist.md)*   |
| Coupled components      | R04.1, P13.2         |


[^1]: Note the ID convention shown is arbitrary — we use `R` for research, `02` for some index on the team's projects, and `1` for a version of the project.

Seismic Analysis for Induced Forecasts

### Top-level requirements

1. The CRS model shall run efficiently and reliably as a module in the broader industrial control system.
2. The optimization module shall continuously validate and deploy both production and shadow instances.
3. The CRS algorithm and broader optimization module shall test for, be robust against, and log and shifts in data, environment, or other operations.
4. The control system optimzation scheme shall be resilient to faults and intrusions (internally and externally).
5. The optimization module shall have a total runtime less than 5% of the end-to-end control software pipeline.
6. The optimization module shall have a fallback/bypass mode in case of unforseen failure.



**Extra notes**: Req's are intentionally succinct and specific to the mixed-variable alg variant we're developing, ignoring generic BO items that are well-studied/validated.


### Model / algorithm info

The CRS Model (Kroll et. al 2017) provides analytical solutions for both the instantaneous earthquake rate following a stress step, in addition to the interseismic rate as a function of time.  The inputs of this model are the pressure p, and its first derivative dp/dt, as functions of time. 

Implementation notes:

The first step is to identify the main shocks, for which we use a peak detector.  Secondly, on the interseismic regions, we can then fit the other CRS parameters. We then fit the mainshock parameters when the rate is highest and cross check if the parameters are in valid bounds. Finally, at test time, our forecasts are generated using the interseismic equation.


### Intended use

At just the software level there are a number of improvements to be made by vectorizing the code, running the grid-search more efficiently using broadcasting. To achieve that, we leverage the pytorch ecosystem, which has the added benefit of automatically calculating gradients for the parameters. This enables us to explore a broader collection of optimizers, many of which incorporate constraints, allowing us to avoid the grid-search entirely.


### Testing status

- Low-level tests verify the algorithm can find solutions for simple mixed-variable functions
- Tests verify the algorithm converges for standard BO problems
- There are several unit tests on the BO loop, but more are needed (notably for diverse parameter sets)
- MVBO algorithm converges to solution on optimization benchmark problems in 1.0s or less on 4-core CPU.

**Extra notes**: Base BO tests are assumed valid from the source BoTorch and GPyTorch repositories


### Data considerations

The data used here are uploaded to Zenodo (https://zenodo.org/record/6957214#.Yuss1ezMJTY)

### Caveats, known edge cases, recommendations

- Forse use local parameters as your input. The threshold of peak detector is used to control overpredict or underpredict most BO problems you are best off starting with default algorithms (see BoTorch)


### MLTRL stage debrief

<!-- Succinct summary of stage progress – please respond to each question, link to extended material if needed... -->

1. What was accomplished, and by who?

   We employed numerical-computing best practices to construct a high-performing CRS model in PyTorch. Through a reimplementation of the CRS model and modifying the optimization algorithm, we have reduced training time from 22 hours to 3 minutes. This makes the CRS model a real-time forecast. This is one of the main pillars of the DOE’s Smart Initiative to move toward large-scale CO2 Sequestration.See *link to experiments page w/ plots*. By Giuseppe Castiglione, Machine Learning Researcher at Frontier Development Lab, USA (R)
Alexandre Chen, University of Oregon, Department of Computer and Information Science, USA (R)
Akshay Suresh, Cornell University, Department of Astronomy, USA (R)
Han Xiao, University of California Santa Barbara, Department of Earth Science, USA (R)
Kayla Kroll, Lawrence Livermore National Laboratory, USA (F)
Christopher Sherman, Lawrence Livermore National Laboratory, USA (F)
Constantin Weisser, QuantumBlack, AI by McKinsey, Boston USA (F)
 

2. What was punted and/or de-scoped?

    n/a

3. What was learned?

One of the main reasons the reference code used grid search is that applying different optimizers directly may generate non-physical settings for the parameters. This can be accommodated by applying constraints to the optimization problem. For example, it is known from the literature that parameters such as seismic rate factors lie within specific physical ranges determined by laboratory experiments. These bounds are easily accommodated by different parameterizations.

We took advantage of several opportunities to reduce the dimensionality of the search space. For example, some parameter pairs - such as 𝛼 and 𝜇 - never appear independently in code. Furthermore, some parameters - such as 𝛥CFS - only affect a specific subset of the full input time series and may be decoupled from other parameters. In summary, the size of the search space may be significantly reduced, further improving efficiency. We will demonstrate that this simplification is possible by conducting a series of small experiments to ensure that all parameters obtained in this way are still within the physical range.

4. What tech debt what gained? Mitigated?

We so that it doesn't require a PhD-trained person to run it. And the operating environment is very simple, and it can run quickly on a personal computer. Has good transferability
