# Unstructured C++ utils

## Interface for generated code (unstructured)

_Allocation_ of a computation object is provided by free function with the following properties

- takes one argument _Mesh_
- returns a Callable _Computation_

_Mesh_ is a `gridtools::hymap` with

- keys representing the connectivities are expressed with a _NeighborChain_
- values model the _Connectivity_ concept.

_NeighborChain_ is

- a _tuple-like_ (see GridTools) object with _LocationTypes_, e.g. `std::tuple<Edge, Node>` (edge to node). \[Potentially, we could model more complex neighbor chains with this approach, e.g. Node->Cell->Edge\]

_LocationTypes_ are

- `struct Node;`
- `struct Edge;`
- `struct Cell;`

_Connectivity_

- `getNeighborTable(Connectivity)` returns a two dimensional SID with dimensions `primary` and `neighbor` (TODO better names)
- `getMaxNeighbors(Connectivity)` returns an `std::integral_constant<std::size_t, N>` with `N` representing the maximal number of neighbors.

_Callable_ has

- one argument for each input or output variable modelling SID
- the unstructured dimension is identified with key _LocationType_ \[Question: how are dimensions identified in GridTools?\]
- a possible sparse dimension is identified with a _NeighborChain_
