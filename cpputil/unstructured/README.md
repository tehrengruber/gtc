# Unstructured C++ utils

## Interface for generated code (unstructured)

###  _Allocation_ of a computation object is provided by free function with the following properties

- takes 2 arguments _Mesh_ and _VerticalIterationSpace_
- returns a Callable _Computation_

### _Mesh_

- `mesh::connectivity<NeighborChain>(Mesh)` returns an object modeling the _Connectivity_ concept.
- `mesh::iteration_space<Tag>` returns a _RegularIterationSpace_ when Tag is a _LocationType_ or a _CustomIterationSpace_ in other cases, e.g. for custom tags like `pole_edges`.

Notes:

- Only Neighbor Chains which are used in the computation need to be available.
- If a `NeighborChain` is present, we can provide a default implementation for the first element of the neighbor chain, e.g. if  `mesh::connectivity<std::tuple<edge, vertex>>(mesh)` is available, we can provide `mesh::connectivity<edge>>(mesh)` for free
- Mesh has a default implementation using hymap


### _NeighborChain_ is

- a compile-time-list (e.g. gridtools::meta::list) with _LocationTypes_, e.g. `std::tuple<edge, vertex>` (edge to vertex).

### _LocationTypes_ are the following tag types

- `struct vertex;`
- `struct edge;`
- `struct cell;`

### _Connectivity_

All return values of the following concept functions need to be valid on device (especially the SID neighbor table)

The following functions are defined
- TODO remove: `connectivity::size(Connectivity)` returns the number of elements of the primary location (as `std::size_t`).  \[Consider compile time sizes as well. TODO this information is also encoded in the connectivities, e.g. the upper_bound of the primary dimension of a neighbor table\]
- `connectivity::max_neighbors(Connectivity)` returns a convertible to int (e.g. `gridtools::integral_constant<int, N>` with `N` representing the maximal number of neighbors, or just `int`)
- `connectivity::skip_value(Connectivity)` returns the element signaling a non-existent value in a regular neighbor table (e.g. of type `int` or `integral_constant`)
- `connectivity::neighbor_table(Connectivity)` returns a two dimensional SID with dimensions _LocationType_ and `neighbor` (TODO better name). Note that the neighbor table SID can be a special SID providing strided access ("on-the-fly neighbor table").

### _RegularIterationSpace_

is a pair of convertible-to-ints, e.g. `int` or `integral_constant<int, N>`, representing the lower and upper bounds of the index space.

### _CustomIterationSpace_

is expected to be a 1 dimensional SID containing the indices to iterate over.

2 SID implementations are useful for the following patterns
- for a range of consecutive indices: this would be implemented by a special SID containing start and endpoint (const ints) and the current position (int).
- set of non-consecutive indices: implemented as a normal array of indices.

### _VerticalIterationSpace_

is a tuple of convertible-to-ints defining sizes of intervals (or maybe splitter positions?).

### _Computation_ has

- one argument for each input or output variable modelling _uSID_

### _unstructured SID (uSID)_
is a SID with a given set of tags to identify dimensions:
- the unstructured dimension is identified with key _LocationType_
- a possible sparse dimension is identified with the `neighbor` tag
  \[TODO maybe something like `local<LocationType>` or `neighbor<LocationType>`, but then the same for the neighbor table to be able to iterate both with the same tag.\]
- the vertical dimension is identified with `namespace dim {struct k;}` (TODO better ideas?)
- TODO what to do with extra dimensions?
