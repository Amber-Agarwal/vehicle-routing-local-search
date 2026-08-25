# Disaster Relief Helicopter Routing

A local-search solver for a multi-trip, multi-depot vehicle routing problem, built for **COL333
(Artificial Intelligence), Assignment 1** at IIT Delhi. The assignment's goal was to take an
open-ended optimization problem, formulate it as a search problem, and solve it — without using
any off-the-shelf search/optimization library. Everything here (state representation, neighborhood
moves, acceptance criterion) is written from scratch.

The full problem statement is in [`docs/A1.pdf`](docs/A1.pdf); the summary below is enough to
follow the code.

## The problem

A flood has cut off a set of villages, each with a known population stranded and known 2D
coordinates. A fleet of helicopters, each based out of a city, must ferry relief packages (dry
food, perishable food, "other" supplies — each with its own weight and value) from cities to
villages.

Each helicopter can make multiple trips. A trip starts and ends at the helicopter's home city,
cannot exceed the helicopter's per-trip weight or distance capacity, and can drop packages at
several villages along the way. A helicopter's *total* distance across all its trips is also
capped. Every trip costs a fixed takeoff/landing fee plus a per-km fuel cost.

The objective is:

```
maximize   (value of relief delivered)  -  (total trip cost)
```

subject to per-trip weight/distance caps, per-helicopter total-distance caps, and a diminishing
return per village once it has received ~9 meals and ~1 unit of other supplies per stranded
person (extra deliveries beyond that add no value). A solution is a full plan for every
helicopter: how many trips, what each trip picks up, which villages it visits and in what order,
and how much is dropped where.

This is a combined **resource allocation + multi-vehicle routing** problem — it doesn't decompose
into an easy assignment or a plain TSP, which is what makes it a good search/optimization exercise.

## Approach: Adaptive Large Neighborhood Search (ALNS)

`solver.cpp` implements ALNS with a simulated-annealing acceptance criterion:

1. **Construction** — six initial solutions are built, one per permutation of the three package
   types (`d`, `p`, `o`). Trying every pickup priority order avoids committing early to one that
   happens to work badly for a given input, and the total time budget is split evenly across the
   six runs (best of the six is kept).
2. **Destroy operators** (remove a batch of stops from the current solution and drop the packages
   back into a shared pool):
   - `random_stop` — removes uniformly random stops
   - `route_remove` — removes an entire trip
   - `shaw` — removes stops that are geographically/structurally related (Shaw-style correlated
     removal), so the repair step has a meaningful sub-problem to reoptimize
   - `worst_values_destroyed` — removes the stops contributing the least value per unit cost
   - `perishable_punished` — targets stops that are mishandling perishable (wet food) allocation
3. **Repair operators** (reinsert pooled packages back into trips):
   - `greedy_insert` — cheapest feasible insertion
   - `cluster_build` — builds new trips by clustering nearby villages
   - `random_insert` — randomized insertion for diversification
   - `new_insert` (regret-style quantity adjustment) / `repair_demand` — rebalances package
     quantities against remaining village demand
4. **Selection** — at each iteration a destroy and a repair operator are picked, applied, and the
   resulting solution is accepted via a Metropolis criterion (`accept if better, else with
   probability exp(Δ/T)`), with `T` cooling over the run. Operator/permutation choices are
   score-weighted based on past success (best solution found / improving / merely accepted).
5. The best solution seen across all six runs is converted to the required output format.

`format_checker.cpp` independently re-parses an output file, validates every constraint (capacity,
distance, trip structure), and reports the objective value — useful both as a submission sanity
check and as a spec for what "feasible" means.

## Repository layout

```
main.cpp             Entry point: reads input, calls solve(), writes output, enforces the time limit
solver.h / solver.cpp ALNS solver (state, destroy/repair operators, acceptance loop)
structures.h          Core data types (Point, Village, Helicopter, Trip, Solution, ...)
io_handler.h/.cpp     Input parsing and output writing per the assignment's file format
format_checker.cpp    Standalone validator: checks a solution's feasibility and prints its score
Makefile              Builds `main` and `format_checker`
docs/A1.pdf           Full assignment specification
sample_io/            A few small, worked input/output examples
tests/inputs/         A range of small/medium/large generated test instances
tests/benchmark.py    Interactive CLI to compile, run and score a batch of test instances
writeup.txt           Submission writeup (collaboration disclosure, as originally submitted)
```

## Building and running

Requires a C++17 compiler (developed against g++ 9.4.0 / Ubuntu 20.04, also builds with any
recent g++).

```bash
make              # builds ./main
make checker      # builds ./format_checker

./main <input_file> <output_file>
./format_checker <input_file> <output_file>   # validates output and prints the objective value
```

Try it on a bundled example:

```bash
./main sample_io/input1.txt out.txt
./format_checker sample_io/input1.txt out.txt
```

## Input / output format

**Input** (one value/line group per line):
1. Time limit, in minutes
2. `DMax` — max total distance per helicopter, in km
3. `w(d) v(d) w(p) v(p) w(o) v(o)` — weight and value per package type
4. `C` followed by `2C` numbers — city coordinates
5. `V` followed by `3V` numbers — village `x y n` (coordinates, people stranded)
6. `H` followed by `5H` numbers — per helicopter: home city id, weight capacity, distance capacity, fixed cost `F`, fuel cost `alpha`

**Output**: one row per helicopter (`helicopter_id, num_trips`), then one row per trip
(pickup quantities, number of villages visited, and `village_id, drops...` per village), then `-1`
to close that helicopter's block. See `docs/A1.pdf` (page 3) for a fully worked example, or read
`format_checker.cpp` for the exact parsing rules.

## Benchmarking

`tests/benchmark.py` is a small interactive runner (`pip install blessed`) that compiles the
project, runs a selectable subset of `tests/inputs/*.txt`, scores each with `format_checker`, and
prints a summary table:

```bash
cd tests
python benchmark.py
```

## Notes

This was originally a two-person team submission; `writeup.txt` is kept as submitted, including
its collaboration disclosure. Per the assignment's rules, LLMs were used only for debugging
assistance on already-written code, not for generating the solution — the search/optimization
logic itself (state design, ALNS operators, acceptance criterion) is original coursework.

## License

[MIT](LICENSE)
