# Unstructured C++ utils

## Interface for generated code (unstructured)

_Allocation_ of a computation object is provided by free function with the following properties

- takes one argument _Mesh_
- returns a Callable _Computation_

_Mesh_

- `mesh::connectivity<NeighborChain>(Mesh)` returns an object modeling the _Connectivity_ concept
<!-- - `mesh::get_size<LocationType>(Mesh)` returns the number of elements of this LocationType (as `std::size_t`) \[Consider compile time sizes as well.\] -->
- \[Note: Mesh has a default implementation using hymap in unstructured_helper.hpp\]

_NeighborChain_ is

- a _tuple-like_ (see GridTools) object with _LocationTypes_, e.g. `std::tuple<edge, vertex>` (edge to vertex). \[Potentially, we could model more complex neighbor chains with this approach, e.g. vertex->cell->edge\]
- \[TODO: tuple-like or `std::tuple`?\]

_LocationTypes_ are the following tag types

- `struct vertex;`
- `struct edge;`
- `struct cell;`

_Connectivity_

- `connectivity::neighbor_table(Connectivity)` returns a two dimensional SID with dimensions _LocationType_ and `neighbor` (TODO better name)
- `connectivity::max_neighbors(Connectivity)` returns a `std::integral_constant<std::size_t, N>` with `N` representing the maximal number of neighbors.
- `connectivity::primary_size(Connectivity)` returns the number of elements of the primary location (as `std::size_t`).  \[Consider compile time sizes as well. TODO this information is also encoded in the connectivities, e.g. the upper_bound of the primary dimension of a neighbor table\]
- `connectivity::skip_value(Connectivity)` returns the element signaling a non-existent value in a rectangular neighbor table
- Connectivity needs to be copyable to device and elements need accessible from device (e.g. the neighbor table)

_Callable_ has

- one argument for each input or output variable modelling SID
- the unstructured dimension is identified with key _LocationType_
- a possible sparse dimension is identified with a _NeighborChain_ (TODO try implementation)

### TODOs

- Need to describe how to skip a value if #neighbors < max_neighbors (some magic number probably)
