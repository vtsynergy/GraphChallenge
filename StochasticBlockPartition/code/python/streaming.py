
import os

from partition_baseline_support import load_graph

def load_graph_parts(input_filename, args):
    true_partition_available = True
    if not os.path.isfile(input_filename + '.tsv') and not os.path.isfile(input_filename + '_1.tsv'):
            print("File doesn't exist: '{}'!".format(input_filename))
            sys.exit(1)

    if args.parts >= 1:
            print('\nLoading partition 1 of {} ({}) ...'.format(args.parts, input_filename + "_1.tsv"))
            out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available, strm_piece_num=1)
            for part in range(2, args.parts + 1):
                    print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
                    out_neighbors, in_neighbors, N, E = load_graph(input_filename, load_true_partition=False, strm_piece_num=part, out_neighbors=out_neighbors, in_neighbors=in_neighbors)
    else:
        if true_partition_available:
            out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available)
        else:
            out_neighbors, in_neighbors, N, E = load_graph(input_filename, load_true_partition=true_partition_available)
            true_partition = None

    return out_neighbors, in_neighbors, N, E, true_partition


def naive_streaming(args):
    input_filename = args.input_filename
    # Emerging edge piece by piece streaming.
    # The assumption is that unlike parallel decimation, where a static graph is cut into
    # multiple subgraphs which do not have the same nodes, the same node set is potentially
    # present in each piece.
    #
    out_neighbors,in_neighbors = None,None
    t_all_parts = 0.0

    for part in range(1, args.parts + 1):
        print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
        t_part = 0.0

        if part == 1:
            out_neighbors, in_neighbors, N, E, true_partition = \
                    load_graph(input_filename,
                               load_true_partition=1,
                               strm_piece_num=part,
                               out_neighbors=None,
                               in_neighbors=None)
        else:
            out_neighbors, in_neighbors, N, E = \
                    load_graph(input_filename,
                               load_true_partition=0,
                               strm_piece_num=part,
                               out_neighbors=out_neighbors,
                               in_neighbors=in_neighbors)

        # Run to ground.
        print('Running partition for part %d N %d E %d' % (part,N,E))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 0, alg_state = None)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        t_all_parts += t_part

        if part == args.parts:
            print('Evaluate final partition.')
        else:
            print('Evaluate part %d' % part)

        precision,recall = evaluate_partition(true_partition, partition)
        print('Elapsed compute time for part %d is %f cumulative %f precision %f recall %f' % (part,t_part,t_all_parts,precision,recall))

    return t_all_parts


def copy_alg_state(alg_state):
    # Create a deep copy of algorithmic state.
    (hist, num_blocks, overall_entropy, partition, interblock_edge_count,block_degrees_out,block_degrees_in,block_degrees,golden_ratio_bracket_established,delta_entropy_threshold,num_blocks_to_merge,optimal_num_blocks_found,n_proposals_evaluated,total_num_nodal_moves) = alg_state

    (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) = hist

    hist_copy = tuple((i.copy() for i in hist))
    try:
        num_blocks_copy = num_blocks.copy()
    except AttributeError:
        num_blocks_copy = num_blocks
    overall_entropy_copy = overall_entropy.copy()
    partition_copy = partition.copy()
    interblock_edge_count_copy = interblock_edge_count.copy()
    block_degrees_out_copy = block_degrees_out.copy()
    block_degrees_in_copy = block_degrees_in.copy()
    block_degrees_copy = block_degrees.copy()
    golden_ratio_bracket_established_copy = golden_ratio_bracket_established # bool
    delta_entropy_threshold_copy = delta_entropy_threshold # float
    num_blocks_to_merge_copy = num_blocks_to_merge # int
    optimal_num_blocks_found_copy = optimal_num_blocks_found # bool
    n_proposals_evaluated_copy = n_proposals_evaluated # int
    total_num_nodal_moves_copy = total_num_nodal_moves # int


    alg_state_copy = (hist_copy, num_blocks_copy, overall_entropy_copy, partition_copy, interblock_edge_count_copy, block_degrees_out_copy, block_degrees_in_copy, block_degrees_copy, golden_ratio_bracket_established_copy, delta_entropy_threshold_copy, num_blocks_to_merge_copy, optimal_num_blocks_found_copy, n_proposals_evaluated_copy, total_num_nodal_moves_copy)

    return alg_state_copy


def incremental_streaming(args):
    input_filename = args.input_filename
    # Emerging edge piece by piece streaming.
    # The assumption is that unlike parallel decimation, where a static graph is cut into
    # multiple subgraphs which do not have the same nodes, the same node set is potentially
    # present in each piece.
    #
    out_neighbors,in_neighbors,alg_state = None,None,None
    t_all_parts = 0.0

    for part in range(1, args.parts + 1):
        t_part = 0.0

        if part == 1:
            print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))

            out_neighbors, in_neighbors, N, E, true_partition = \
                    load_graph(input_filename,
                               load_true_partition=1,
                               strm_piece_num=part,
                               out_neighbors=None,
                               in_neighbors=None)
            min_number_blocks = N / 2
        else:
            # Load true_partition here so the sizes of the arrays all equal N.
            if alg_state:
                print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))

                out_neighbors, in_neighbors, N, E, alg_state,t_compute = \
                                                load_graph(input_filename,
                                                           load_true_partition=1,
                                                           strm_piece_num=part,
                                                           out_neighbors=out_neighbors,
                                                           in_neighbors=in_neighbors,
                                                           alg_state = alg_state)
                t_part += t_compute
                print("Intermediate load_graph compute time for part %d is %f" % (part,t_compute))
                t0 = timeit.default_timer()
                hist = alg_state[0]
                (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) = hist

                print("Incrementally updated alg_state for part %d" %(part))
                print('New Overall entropy: {}'.format(old_overall_entropy))
                print('New Number of blocks: {}'.format(old_num_blocks))
                print("")

                verbose = 1
                n_thread = args.threads
                batch_size = args.node_move_update_batch_size
                vertex_num_in_neighbor_edges = np.empty(N, dtype=int)
                vertex_num_out_neighbor_edges = np.empty(N, dtype=int)
                vertex_num_neighbor_edges = np.empty(N, dtype=int)
                vertex_neighbors = [np.concatenate((out_neighbors[i], in_neighbors[i])) for i in range(N)]

                for i in range(N):
                    vertex_num_out_neighbor_edges[i] = sum(out_neighbors[i][:,1])
                    vertex_num_in_neighbor_edges[i] = sum(in_neighbors[i][:,1])
                    vertex_num_neighbor_edges[i] = vertex_num_out_neighbor_edges[i] + vertex_num_in_neighbor_edges[i]
                #delta_entropy_threshold = delta_entropy_threshold1 = 5e-4
                delta_entropy_threshold = 1e-4

                for j in [0,2,1]:
                    if old_interblock_edge_count[j] == []:
                        continue

                    print("Updating previous state in bracket history.")

                    M_old = old_interblock_edge_count[j].copy()
                    M = old_interblock_edge_count[j]
                    partition = old_partition[j]
                    block_degrees_out = old_block_degrees_out[j]
                    block_degrees_in = old_block_degrees_in[j]
                    block_degrees = old_block_degrees[j]
                    num_blocks = old_num_blocks[j]
                    overall_entropy = old_overall_entropy[j]

                    total_num_nodal_moves_itr = nodal_moves_parallel(n_thread, batch_size, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose, args)

                t1 = timeit.default_timer()
                print("Intermediate nodal move time for part %d is %f" % (part,(t1-t0)))
                t_part += (t1 - t0)
            else:
                # We are not doing partitioning yet. Just wait.
                out_neighbors, in_neighbors, N, E, true_partition = \
                                                load_graph(input_filename,
                                                           load_true_partition=1,
                                                           strm_piece_num=part,
                                                           out_neighbors=out_neighbors,
                                                           in_neighbors=in_neighbors,
                                                           alg_state = None)

            print("Loaded piece %d N %d E %d" % (part,N,E))
            min_number_blocks = int(min_number_blocks / 2)

        print('Running partition for part %d N %d E %d and min_number_blocks %d' % (part,N,E,min_number_blocks))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition,alg_state = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 1, alg_state = alg_state, min_number_blocks = min_number_blocks)
        min_number_blocks /= 2

        alg_state_copy = copy_alg_state(alg_state)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        print("Intermediate partition until save point for part %d is %f" % (part,(t1-t0)))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 0, alg_state = alg_state_copy, min_number_blocks = 5)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        print("Intermediate partition until completion for part %d is %f" % (part,(t1-t0)))

        print('Evaluate part %d' % (part))
        precision,recall = evaluate_partition(true_partition, partition)

        t_all_parts += t_part
        print('Elapsed compute time for part %d is %f cumulative %f precision %f recall %f' % (part,t_part,t_all_parts,precision,recall))

    return t_all_parts
