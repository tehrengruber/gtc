# Unstructured C++ utils

## Interface for generated code (unstructured)

_Allocation_ of a computation object is provided by free function with the following properties

- takes one argument _Mesh_
- returns a Callable _Computation_

_Mesh_ is a `gridtools::hymap` with

- keys representing the connectivities are expressed with a _NeighborChain_
- values model the _Connectivity_ concept.

_NeighborChain_ is

<!-- - a _tuple-like_ (see GridTools) object with _LocationTypes_, e.g. `std::tuple<edge, vertex>` (edge to vertex). \[Potentially, we could model more complex neighbor chains with this approach, e.g. vertex->cell->edge\] -->
- a `std::tuple` with _LocationTypes_, e.g. `std::tuple<edge, vertex>` (edge to vertex). \[Potentially, we could model more complex neighbor chains with this approach, e.g. vertex->cell->edge\] -->

_LocationTypes_ are the following tags

- `struct vertex;`
- `struct edge;`
- `struct cell;`

_Connectivity_

- `get_neighbor_table(Connectivity)` returns a two dimensional SID with dimensions _LocationType_ and `neighbor` (TODO better name)
- `get_max_neighbors(Connectivity)` returns an `std::integral_constant<std::size_t, N>` with `N` representing the maximal number of neighbors.

_Callable_ has

- one argument for each input or output variable modelling SID
- the unstructured dimension is identified with key _LocationType_ \[Question: how are dimensions identified in GridTools?\]
- a possible sparse dimension is identified with a _NeighborChain_
