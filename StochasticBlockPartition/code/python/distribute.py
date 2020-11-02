
from typing import List, Tuple
import sys

import numpy as np

# GraphChallenge code
from partition_baseline_support import carry_out_best_merges
from partition_baseline_support import compute_delta_entropy
from partition_baseline_support import compute_new_block_degrees
from partition_baseline_support import compute_new_rows_cols_interblock_edge_count_matrix
from partition_baseline_support import initialize_edge_counts

# iHeartGraph code
from fast_sparse_array import take_nonzero


class Sample():
    def __init__(self, start: int, out_neighbors: List[List[Tuple[int, int]]]):
        self.vertices = list()  # type: List[int]
        self.vertices.append(start)
        # self.neighbors = list()  # type: List[Tuple[int, int]]
        self.neighbors = out_neighbors[start]
        self.neighborhood = [False] * len(out_neighbors)
        for neighbor in self.neighbors:
            self.neighborhood[neighbor[0]] = True
        self.vertex_map = dict()  # type: Dict[int, int]
        self.vertex_map[start] = 0
    # End of init()

    def pick_vertex(self, sampled_vertices: List[bool], out_neighbors: List[List[Tuple[int, int]]]) -> bool:
        remove_indices = list()  # type: List[int]
        picked = False
        picked_random = False
        for index, neighbor in enumerate(self.neighbors):
            if sampled_vertices[neighbor[0]] is True:  # If already sampled, continue
                remove_indices.append(index)
                continue
            else:  # If not, add vertex to sample
                vertex = neighbor[0]
                # print("picked vertex: ", vertex)
                self.vertex_map[vertex] = len(self.vertices)
                self.vertices.append(vertex)
                for potential_neighbor in out_neighbors[vertex]:
                    potential_vertex = potential_neighbor[0]
                    if self.neighborhood[potential_vertex] is False and sampled_vertices[potential_neighbor] is False:
                        self.neighbors.append(potential_neighbor)
                        self.neighborhood[potential_vertex] = True
                remove_indices.append(index)
                sampled_vertices[vertex] = True
                picked = True
                break
        if not picked:  # if unsampled neighborhood is empty, pick random unsampled vertex
            picked_random = True
            indices = np.where(sampled_vertices == False)[0]
            # print("idx: ", indices)
            vertex = np.random.choice(indices)
            # print("neighborhood was empty, picked random vertex: ", vertex)
            self.vertex_map[vertex] = len(self.vertices)
            self.vertices.append(vertex)
            sampled_vertices[vertex] = True
            for potential_neighbor in out_neighbors[vertex]:
                potential_vertex = potential_neighbor[0]
                if self.neighborhood[potential_vertex] is False and sampled_vertices[potential_neighbor] is False:
                    self.neighbors.append(potential_neighbor)
                    self.neighborhood[potential_vertex] = True

        # remove sampled vertices from list of neighbors
        mask = np.ones(len(self.neighbors), np.bool)
        mask[remove_indices] = 0
        self.neighbors = self.neighbors[mask]
        # for index in remove_indices:  # remove sampled vertices from list of neighbors
        #     neighbor = self.neighbors[index]
        #     self.neighborhood[neighbor[0]] = False
        #     # cannot delete this, so have to use indexing trick. Probably a good place for optimization
        #     mask = np.ones(len(data), np.bool)
        #     mask[sample_indexes] = 0
        #     other_data = data[mask]
        #     self.neighbors = self.neighbors[:index]
        #     self.neighbors = self.neighbors[index + 1:]
        #     # del self.neighbors[index]
        return picked_random
    # End of pick_vertex()
# End of Sample()


def decimate_graph(out_neighbors, in_neighbors, true_partition, decimation, decimated_piece):
    """
    """
    sampled = np.arange(len(in_neighbors))[decimated_piece::decimation]
    # print("sampled.size = ", sampled.size)

    in_neighbors = in_neighbors[decimated_piece::decimation]
    out_neighbors = out_neighbors[decimated_piece::decimation]
    true_partition = true_partition[decimated_piece::decimation]
    E = sum(len(v) for v in out_neighbors)
    N = np.int64(len(in_neighbors))

    for i in range(N):
        xx = (in_neighbors[i][:, 0] % decimation) == decimated_piece
        in_neighbors[i] = in_neighbors[i][xx, :]
        xx = (out_neighbors[i][:, 0] % decimation) == decimated_piece
        out_neighbors[i] = out_neighbors[i][xx, :]

    for i in range(N):
        in_neighbors[i][:,0] = in_neighbors[i][:,0] / decimation
        out_neighbors[i][:,0] = out_neighbors[i][:,0] / decimation

    return out_neighbors, in_neighbors, N, E, true_partition, sampled


def decimate_graph_snowball(out_neighbors, in_neighbors, true_partition, decimation, decimated_piece):
    """TODO
    """
    # Do snowball for ALL parts (can calculate with decimation value?)
    # Use decimated_piece to select the appropriate snowball sample
    # If one snowball piece gets stuck, add random pieces? Continue snowball from unsampled piece?
    num_vertices = len(true_partition)
    sampled_vertices = np.asarray([False] * num_vertices)
    np.random.seed(1000)
    starting_vertices = np.random.choice(np.arange(num_vertices), size=decimation, replace=False)
    sampled_vertices[starting_vertices] = True
    # Each sample should have the following:
    samples = [Sample(vertex, out_neighbors) for vertex in starting_vertices]
    num_sampled = len(starting_vertices)
    num_randoms = 0
    while num_sampled < num_vertices:
        for sample in samples:
            if num_sampled == num_vertices:
                break
            # num_left = np.where(sampled_vertices == False)[0].size
            # actual_num_left = num_vertices - num_sampled
            # if num_left != actual_num_left:
            #     print("num sampled: ", num_sampled)
            #     print("num left is: ", np.where(sampled_vertices == False)[0].size, " should be: ", num_vertices - num_sampled)
            #     exit()
            picked_random = sample.pick_vertex(sampled_vertices, out_neighbors)
            if picked_random:
                num_randoms += 1
            num_sampled += 1
        if num_sampled == num_vertices:
            break
    print("num random picks = {} / {}".format(num_randoms, num_vertices))

    # Build sampled graph
    sample_out_neighbors = list()  # type: List[List[Tuple[int, int]]]
    sample_in_neighbors = list()  # type: List[List[Tuple[int, int]]]
    vertex_list = samples[decimated_piece].vertices
    for index in vertex_list:
        sampled_out_neighbors = out_neighbors[index]
        out_mask = np.isin(sampled_out_neighbors[:, 0], vertex_list, assume_unique=False)
        sampled_out_neighbors = sampled_out_neighbors[out_mask]
        for out_neighbor in sampled_out_neighbors:
            try:
                out_neighbor[0] = samples[decimated_piece].vertex_map[out_neighbor[0]]
            except KeyError as e:
                print()
                print(samples[decimated_piece].vertex_map)
                print()
                print()
                raise e
        sample_out_neighbors.append(sampled_out_neighbors)
        sampled_in_neighbors = in_neighbors[index]
        in_mask = np.isin(sampled_in_neighbors[:, 0], vertex_list, assume_unique=False)
        sampled_in_neighbors = sampled_in_neighbors[in_mask]
        for in_neighbor in sampled_in_neighbors:
            in_neighbor[0] = samples[decimated_piece].vertex_map[in_neighbor[0]]
        sample_in_neighbors.append(sampled_in_neighbors)
        # self.num_edges += np.sum(out_mask) + np.sum(in_mask)
    # true_block_assignment = old_true_block_assignment[sampled_vertices]
    # true_blocks = list(set(true_block_assignment))
    # self.true_blocks_mapping = dict([(v, k) for k,v in enumerate(true_blocks)])
    # self.true_block_assignment = np.asarray([self.true_blocks_mapping[b] for b in true_block_assignment])
    sample_true_partition = true_partition[vertex_list]
    # self.sample_num = len(self.vertex_mapping)

    # in_neighbors = in_neighbors[decimated_piece::decimation]
    # out_neighbors = out_neighbors[decimated_piece::decimation]
    # true_partition = true_partition[decimated_piece::decimation]
    E = sum(len(v) for v in sample_out_neighbors)
    N = np.int64(len(sample_in_neighbors))

    # TODO: convert vertex numbers to 0 - N. Maybe not needed?

    # for i in range(N):
    #     xx = (in_neighbors[i][:,0] % decimation) == decimated_piece
    #     in_neighbors[i] = in_neighbors[i][xx, :]
    #     xx = (out_neighbors[i][:,0] % decimation) == decimated_piece
    #     out_neighbors[i] = out_neighbors[i][xx, :]

    # for i in range(N):
    #     in_neighbors[i][:,0] = in_neighbors[i][:,0] / decimation
    #     out_neighbors[i][:,0] = out_neighbors[i][:,0] / decimation

    return sample_out_neighbors, sample_in_neighbors, N, E, sample_true_partition, vertex_list
# decimate_graph_snowball()


def merge_partitions(partitions, stop_pieces, out_neighbors, verbose, use_sparse_alg, use_sparse_data):
    """
    Create a unified graph block partition from the supplied partition pieces into a partiton of size stop_pieces.
    """

    pieces = len(partitions)
    N = sum(len(i) for i in partitions)

    # The temporary partition variable is for the purpose of computing M.
    # The axes of M are concatenated block ids from each partition.
    # And each partition[i] will have an offset added to so all the interim partition ranges are globally unique.
    #
    partition = np.zeros(N, dtype=int)

    while pieces > stop_pieces:

        # TODO: the out_neighbors are not in the same order as the vertices in partition
        # TODO: fill the partition array (it's currently empty)

        Bs = [max(partitions[i]) + 1 for i in range(pieces)]
        B =  sum(Bs)

        partition_offsets = np.zeros(pieces, dtype=int)
        partition_offsets[1:] = np.cumsum(Bs)[:-1]

        if verbose > 1:
            print("")
            print("Reconstitute graph from %d pieces B[piece] = %s" % (pieces,Bs))
            print("partition_offsets = %s" % partition_offsets)

        print("partition_offsets", partition_offsets)
        print("Bs", Bs)
        # It would likely be faster to re-use already computed values of M from pieces:
        #     M[ 0:B0,     0:B0   ] = M_0
        #     M[B0:B0+B1, B0:B0+B1] = M_1
        # Instead of relying on initialize_edge_counts.
        print(partition)
        M, block_degrees_out, block_degrees_in, block_degrees \
            = initialize_edge_counts(out_neighbors, B, partition, use_sparse_data)
        print(M)

        if verbose > 2:
            print("M.shape = %s, M = \n%s" % (str(M.shape),M))

        next_partitions = []
        for i in range(0, pieces, 2):
            print("Merge piece %d and %d into %d" % (i, i + 1, i // 2))
            partitions[i],_ = resolve_two_partitions(M, block_degrees_out, block_degrees_out, block_degrees_out,
                                                   partitions[i], partitions[i + 1],
                                                   partition_offsets[i], partition_offsets[i + 1],
                                                   Bs[i], Bs[i + 1],
                                                   verbose,
                                                   use_sparse_alg,
                                                   use_sparse_data)
            next_partitions.append(np.concatenate((partitions[i], partitions[i+1])))

        partitions = next_partitions
        pieces //= 2

    return partitions
# End of merge_partitions()


def merge_two_partitions(M, block_degrees_out, block_degrees_in, block_degrees, partition0, partition1, partition_offset_0, partition_offset_1, B0, B1, verbose, use_sparse_alg, use_sparse_data):
    """
    Merge two partitions each from a decimated piece of the graph.
    Note
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    block count matrix between all the blocks, of which partition 0 and partition 1 are just subsets
        partition_offset_0 and partition_offset_1 are the starting offsets within M of each partition piece
    """
    # Now reduce by merging down blocks from partition 0 into partition 1.
    # This requires computing delta_entropy over all of M (hence the partition_offsets are needed).

    delta_entropy = np.empty((B0,B1))

    for r in range(B0):
        current_block = r + partition_offset_0

        # Index of non-zero block entries and their associated weights
        in_idx, in_weight = take_nonzero(M, current_block, 1, sort = False)
        out_idx, out_weight = take_nonzero(M, current_block, 0, sort = False)

        block_neighbors = np.concatenate((in_idx, out_idx))
        block_neighbor_weights = np.concatenate((in_weight, out_weight))

        num_out_block_edges = sum(out_weight)
        num_in_block_edges = sum(in_weight)
        num_block_edges = num_out_block_edges + num_in_block_edges

        for s in range(B1):
            proposal = s + partition_offset_1

            new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col \
                = compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, proposal,
                                                                     out_idx, out_weight,
                                                                     in_idx, in_weight,
                                                                     M[current_block, current_block], agg_move = 1,
                                                                     use_sparse_alg = use_sparse_alg)

            block_degrees_out_new, block_degrees_in_new, block_degrees_new \
                = compute_new_block_degrees(current_block,
                                            proposal,
                                            block_degrees_out,
                                            block_degrees_in,
                                            block_degrees,
                                            num_out_block_edges,
                                            num_in_block_edges,
                                            num_block_edges)

            delta_entropy[r, s] = compute_delta_entropy(current_block, proposal, M,
                                                        new_M_r_row,
                                                        new_M_s_row,
                                                        new_M_r_col,
                                                        new_M_s_col,
                                                        block_degrees_out,
                                                        block_degrees_in,
                                                        block_degrees_out_new,
                                                        block_degrees_in_new)

    best_merge_for_each_block = np.argmin(delta_entropy, axis=1)

    # if verbose == 0:
    #     print("delta_entropy = \n%s" % delta_entropy)
    #     print("best_merge_for_each_block = %s" % best_merge_for_each_block)

    delta_entropy_for_each_block = delta_entropy[np.arange(delta_entropy.shape[0]), best_merge_for_each_block]

    # Global number of blocks (when all pieces are considered together).
    num_blocks = M.shape[0]
    num_blocks_to_merge = B0
    best_merges = delta_entropy_for_each_block.argsort()

    # Note: partition0 will be modified in carry_out_best_merges
    (partition, num_blocks) = carry_out_best_merges(delta_entropy_for_each_block,
                                                    best_merges,
                                                    best_merge_for_each_block + partition_offset_1,
                                                    partition0,
                                                    num_blocks,
                                                    num_blocks_to_merge, verbose=(verbose > 2))

    return partition, num_blocks
# End of merge_two_partitions()


def resolve_partitions(partitions, vertex_lists, stop_pieces, out_neighbors, verbose, use_sparse_alg, use_sparse_data):
    """
    Create a unified graph block partition from the supplied partition pieces into a partiton of size stop_pieces.
    """

    pieces = len(partitions)
    N = sum(len(i) for i in partitions)

    # The temporary partition variable is for the purpose of computing M.
    # The axes of M are concatenated block ids from each partition.
    # And each partition[i] will have an offset added to so all the interim partition ranges are globally unique.
    #
    partition = np.zeros(N, dtype=int)

    while pieces > stop_pieces:

        Bs = [max(partitions[i]) + 1 for i in range(pieces)]
        B = sum(Bs)

        partition_offsets = np.zeros(pieces, dtype=int)
        partition_offsets[1:] = np.cumsum(Bs)[:-1]
        # print("partition_offsets", partition_offsets)
        print("Bs", Bs)

        # TODO: the out_neighbors are not in the same order as the vertices in partition
        start = 0
        for index in range(len(partitions)):
            partition_length = partitions[index].size
            # print(vertex_lists[index])
            # print("vertex_lists[index]: ", vertex_lists[index].size)
            # print("partitions[index]", partitions[index].size)
            partition[vertex_lists[index]] = (partitions[index] + partition_offsets[index])
            start += partition_length

        if verbose > 1:
            print("")
            print("Reconstitute graph from %d pieces B[piece] = %s" % (pieces, Bs))
            print("partition_offsets = %s" % partition_offsets)

        # It would likely be faster to re-use already computed values of M from pieces:
        #     M[ 0:B0,     0:B0   ] = M_0
        #     M[B0:B0+B1, B0:B0+B1] = M_1
        # Instead of relying on initialize_edge_counts.
        # print(partition)
        M, block_degrees_out, block_degrees_in, block_degrees \
            = initialize_edge_counts(out_neighbors, B, partition, use_sparse_data)

        # print("degrees: ", block_degrees_out, block_degrees_in, block_degrees)
        # print(M)

        if verbose > 2:
            print("M.shape = %s, M = \n%s" % (str(M.shape), M))

        next_partitions = []
        next_vertex_lists = []
        for i in range(0, pieces, 2):
            # print("Merge piece %d and %d into %d" % (i, i + 1, i // 2))
            partitions[i], _ = resolve_two_partitions(M, block_degrees_out, block_degrees_out, block_degrees_out,
                                                      partitions[i], partitions[i + 1], partition_offsets[i],
                                                      partition_offsets[i + 1], Bs[i], Bs[i + 1], verbose,
                                                      use_sparse_alg, use_sparse_data)
            next_partitions.append(np.concatenate((partitions[i], partitions[i + 1])))
            next_vertex_lists.append(np.concatenate((vertex_lists[i], vertex_lists[i + 1])))

        partitions = next_partitions
        vertex_lists = next_vertex_lists
        pieces //= 2

    print(partitions)
    return partitions
# End of resolve_partitions()


def resolve_two_partitions(M, block_degrees_out, block_degrees_in, block_degrees, partition0, partition1,
                           partition_offset_0, partition_offset_1, B0, B1, verbose, use_sparse_alg, use_sparse_data):
    """
    Merge two partitions each from a decimated piece of the graph.
    Note
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    block count matrix between all the blocks, of which partition 0 and partition 1 are just subsets
        partition_offset_0 and partition_offset_1 are the starting offsets within M of each partition piece
    """
    # Now reduce by merging down blocks from partition 0 into partition 1.
    # This requires computing delta_entropy over all of M (hence the partition_offsets are needed).

    delta_entropy = np.empty((B0, B1))

    for r in range(B0):
        current_block = r + partition_offset_0

        # Index of non-zero block entries and their associated weights
        in_idx, in_weight = take_nonzero(M, current_block, 1, sort=False)
        out_idx, out_weight = take_nonzero(M, current_block, 0, sort=False)

        num_out_block_edges = sum(out_weight)
        num_in_block_edges = sum(in_weight)
        num_block_edges = num_out_block_edges + num_in_block_edges

        for s in range(B1):
            proposal = s + partition_offset_1

            new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col \
                = compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, proposal, out_idx, out_weight,
                                                                     in_idx, in_weight, M[current_block, current_block],
                                                                     agg_move=1, use_sparse_alg=use_sparse_alg)

            block_degrees_out_new, block_degrees_in_new, block_degrees_new \
                = compute_new_block_degrees(current_block,
                                            proposal,
                                            block_degrees_out,
                                            block_degrees_in,
                                            block_degrees,
                                            num_out_block_edges,
                                            num_in_block_edges,
                                            num_block_edges)

            delta_entropy[r, s] = compute_delta_entropy(current_block, proposal, M, new_M_r_row, new_M_s_row,
                                                        new_M_r_col, new_M_s_col, block_degrees_out, block_degrees_in,
                                                        block_degrees_out_new, block_degrees_in_new)

    best_merge_for_each_block = np.argmin(delta_entropy, axis=1)

    delta_entropy_for_each_block = delta_entropy[np.arange(delta_entropy.shape[0]), best_merge_for_each_block]
    print("de for each block", delta_entropy_for_each_block)

    # Global number of blocks (when all pieces are considered together).
    num_blocks = M.shape[0]
    num_blocks_to_merge = B0
    best_merges = delta_entropy_for_each_block.argsort()
    print(best_merges)

    # Note: partition0 will be modified in carry_out_best_merges
    (partition, num_blocks) = conditional_merge(delta_entropy_for_each_block, best_merges,
                                                best_merge_for_each_block + partition_offset_1, partition0,
                                                num_blocks, num_blocks_to_merge, verbose=(verbose > 2))

    return partition, num_blocks
# End of resolve_two_partitions()


def conditional_merge(delta_entropy_for_each_block, best_merges, best_merge_for_each_block, b, B, B_to_merge,
                      verbose=False):
    """Execute the best merge (agglomerative) moves to reduce a set number of blocks

        Parameters
        ----------
        delta_entropy_for_each_block : ndarray (float)
                    the delta entropy for merging each block
        best_merge_for_each_block : ndarray (int)
                    the best block to merge with for each block
        b : ndarray (int)
                    array of block assignment for each node
        B : int
                    total number of blocks in the current partition
        B_to_merge : int
                    the number of blocks to merge

        Returns
        -------
        b : ndarray (int)
                    array of new block assignment for each node after the merge
        B : int
                    total number of blocks after the merge
    """
    block_map = np.arange(B)
    num_merge = 0

    for counter in range(len(best_merges)):
        if counter == len(best_merges):
            if verbose:
                print("No more merges possible")
            break

        mergeFrom = best_merges[counter]
        mergeTo = block_map[best_merge_for_each_block[best_merges[counter]]]

        if delta_entropy_for_each_block[mergeFrom] > 0:
            if verbose:
                print("Skipping bad merge from {} to {}: de {}".format(
                      mergeFrom, mergeTo, delta_entropy_for_each_block[mergeFrom]))
            continue

        if mergeTo != mergeFrom:
            if verbose:
                print("Merge %d of %d from block %s to block %s" % (num_merge, B_to_merge, mergeFrom, mergeTo))
            block_map[np.where(block_map == mergeFrom)] = mergeTo
            b[np.where(b == mergeFrom)] = mergeTo
            num_merge += 1

    remaining_blocks = np.unique(b)
    mapping = -np.ones(B, dtype=int)
    mapping[remaining_blocks] = np.arange(len(remaining_blocks))
    b = mapping[b]
    B -= num_merge
    return b, B
# End of conditional_merge()
