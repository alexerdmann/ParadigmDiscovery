from sys import stderr, stdout
import os
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from functools import lru_cache
import pickle as pkl
# Top level libraries
from evaluation import evaluate
from inflector import Inflector
# Sub libraries
from Utils.segment_functions import lcs, lcs1, getExponent
from Utils.grid_utils import Averager, get_fasttext_vectors, custom_cluster
from Utils.segment_functions import lcs, getExponent
from Seq2seq import dataloader, seq2seq_runner


class GridManager(object):

    #############################################################
    ### Initialization Functions
    #############################################################
    def __init__(self, D, args, n_neighbors=500):

        stdout.write('Language: {}\n'.format(D.lg))
        stdout.write('Number of cells: {}\n'.format(args.maybe_num_cells))
        stdout.write('Number of paradigms: {}\n'.format(args.maybe_num_paradigms))
        stdout.write('Working directory: {}\n\n\n'.format(args.model_location))
        stdout.flush()

        self.lg, self.model_location = D.lg, os.path.join(args.model_location, D.lg)
        self.full_grid = None  # True once we generate realizations for all empty slots
        # Initialize reference stats
        self.first_empty_row = 0
        self.first_singleton = None
        self.min_exp_prob = 1e-9
        # Store some data
        self.sents = D.sents
        self.gold, self.gold_UM_intersect, self.ext_analogies = D.gold, D.gold_UM_intersect, D.ext_analogies  # just for eval

        # Some initialization options
        self.baseline, self.debug, self.n_neighbors, self.exponent_penalty, self.exponent_penalty_discount = args.baseline, args.debug, n_neighbors, args.exponent_penalty, args.exponent_penalty_discount

        # Potentially restore grid
        if args.restore_grid != None:
            self.restore_grid(args.restore_grid)

        else:
            # More initialization options
            if self.baseline == 'supervised_extrinsic':
                self.run_supervised_extrinsic()
            self.set_r(D.p)
            wf_2_col_idx, col_idx_2_wf = self.set_c(D.wfs, D.c, masked_embeddings=args.masked_embeddings, target_affix_embeddings=args.target_affix_embeddings, target_syntactic_windows=args.target_syntactic_windows)
            # Initialize data structures
            self.wf_grid, self.exp_grid = np.array([[None]*self.c]*self.r), np.array([[None]*self.c]*self.r)  # rXc grids
            self.row_bases, self.col_exp_probs, self.col_ranked_exps  = np.array([None]*self.r), np.array([{}]*self.c), [[]]*self.c  # 1Xr and 1Xc grids

            # Get Initial wf assignments to grid coordinates
            self.initialize_grid(wf_2_col_idx, col_idx_2_wf)

    def restore_grid(self, restore_grid):
        stderr.write('Restoring initialized grid from\n\t{}\n'.format(restore_grid))
        stderr.flush()

        items = pkl.load(open(restore_grid, "rb"))
        self.r, self.c, self.first_singleton, self.first_empty_row, self.wf_grid, self.exp_grid, self.row_bases, self.full_grid, self.col_ranked_exps, self.col_exp_probs = items
        self.get_analyses()
        
    def run_supervised_extrinsic(self):
        # Initialize only the necessary attributes
        self.r, self.c, self.first_empty_row = len(self.gold.lem_2_wf), len(self.gold.cell_2_wf), 0
        self.wf_grid = np.array([[None]*self.c]*self.r)
        # Fill in the grid as if it were supervised
        cells = sorted(list(self.gold.cell_2_wf))
        row_idx = -1
        for lem in self.gold.lem_2_cell_2_wf:
            row_idx += 1
            for cell in self.gold.lem_2_cell_2_wf[lem]:
                col_idx = cells.index(cell)
                self.assign(random.choice(list(self.gold.lem_2_cell_2_wf[lem][cell])), (row_idx, col_idx))
        # Use fully supervised seq2seq to complete the grid
        self.full_grid = self.PCFP()
        self.extrinsic_accuracy, self.extrinsic_accuracy_partial = evaluate(self, eval_fn='extrinsic', msg='Extrinsic Analogical Evaluation of Supervised {} Grid'.format(self.lg))
        exit()

    def set_r(self, maybe_num_paradigms):
        if not isinstance(maybe_num_paradigms, int):
            assert maybe_num_paradigms == 'blind'
            # Grid will only use the paradigms it needs from these 5000 and will add more if it runs out of rows as necessary
            maybe_num_paradigms = 5000
        self.r = maybe_num_paradigms
    
    def set_c(self, wfs, maybe_num_cells, masked_embeddings=True, target_affix_embeddings=True, target_syntactic_windows=True):
        self.c = maybe_num_cells
        # Get initial column clusters from syntactically biased embeddings
        wf_2_col_idx, col_idx_2_wf = self.init_col_clusters(wfs, masked_embeddings=masked_embeddings, target_affix_embeddings=target_affix_embeddings, target_syntactic_windows=target_syntactic_windows)  # will set self.c if self.c == 'blind'
        return wf_2_col_idx, col_idx_2_wf

    def initialize_grid(self, wf_2_col_idx, col_idx_2_wf):

        # Use both embeddings + string features to heuristically initialize grid
        stderr.write('Initializing grid..\n')
        stderr.flush()

        self.PDP(wf_2_col_idx, col_idx_2_wf)
        self.get_analyses()
        self.full_grid = self.PCFP()

        restore_grid = os.path.join(self.model_location, 'restore_me.pkl')
        stderr.write('Pickling out initialized grid to\n\t{}\n'.format(restore_grid))
        stderr.flush()
        items = (self.r, self.c, self.first_singleton, self.first_empty_row, self.wf_grid, self.exp_grid, self.row_bases, self.full_grid, self.col_ranked_exps, self.col_exp_probs)
        pkl.dump(items, open(restore_grid, 'wb'))
        
    def PDP(self, wf_2_col_idx, col_idx_2_wf):

        orig_wf_2_col_idx = dict(wf_2_col_idx)
        orig_col_idx_2_wf = dict((col_idx, set(col_idx_2_wf[col_idx])) for col_idx in range(self.c))

        # Assign initial grid coordinates to each wf
        first_row_to_consider = 0
        for min_col_idx in range(self.c):
            # Start new paradigms for all words not fitting a previous paradigm
            for wf in col_idx_2_wf[min_col_idx]:
                if self.first_empty_row == None:
                    # Updates first_empty_row during assignment
                    self.add_paradigm()
                self.assign(wf, (self.first_empty_row, min_col_idx))
                del wf_2_col_idx[wf]
            del col_idx_2_wf[min_col_idx]

            # For each new singleton-so-far par, consider filling each of the cells to the right of min_col_idx with a yet unassigned form s.t. we maximize  |paradigm|*|lcs(paradigm)| - sum(|affix| for affix in exponent for exponent in exponents)
            for row_idx in range(first_row_to_consider, min(x for x in (self.r, self.first_empty_row) if x)):
                # Take the best option from the final sorted beam
                options, best_score, best_wfs, best_row = self.get_init_row(row_idx, min_col_idx, col_idx_2_wf)
                best_new_wf = None
                for col_idx in range(min_col_idx+1, self.c):
                    best_new_wf = best_row[col_idx]
                    if best_new_wf != None:
                        self.assign(best_new_wf, (row_idx, col_idx))
                        # Once assigned, remove from eligible dictionaries
                        col_idx_2_wf[col_idx].remove(best_new_wf)
                        del wf_2_col_idx[best_new_wf]
                        del self.col_2_wf_2_pot_par_mates[col_idx][best_new_wf]
                self.update_row_seg_analyses(row_idx)

                # Track progress
                if row_idx % 500 == 0:
                    stderr.write('\tinitialization progress: {}%\n'.format(round(100*(row_idx/self.r), 4)))
                    if self.debug:
                        stderr.write('\t\t**singleton and best-non-singleton candidate paradigms**\n') 
                        for ranked_tup in options:
                            stderr.write('\t\tScore: {} \t{}\n'.format(ranked_tup[0], ', '.join(str(x) for x in ranked_tup[2])))    
                    stderr.flush()

            # Update and sanity check
            first_row_to_consider = row_idx + 1
        assert len(wf_2_col_idx) == len(col_idx_2_wf) == 0

        if self.exponent_penalty_discount:
            lcs1.cache_clear()
            self.reinitialize_with_weighted_scores(orig_col_idx_2_wf, orig_wf_2_col_idx)
            lcs1.cache_clear()
            
        # Evaluate grid initialization
        self.joint_sort_rows()
        if self.debug:
            self.debug_print_grid()
    
    def PCFP(self):

        # 1. Open a new sub directory for model-to-be-trained
        ext_model_location = os.path.join(self.model_location, 'Extrinsic_grid_completion')
        if not os.path.isdir(self.model_location):
            os.makedirs(self.model_location)

        # 1.5 Initialize the model
        extrinsic_inflection = Inflector(ext_model_location, data_format=dataloader.DataFormat.MT, extrinsic=True)
        extrinsic_inflection.train = os.path.join(self.model_location, 'ext_train.tsv')
        extrinsic_inflection.dev = os.path.join(self.model_location, 'ext_dev.tsv')

        # 1.75. Write out the initial grid for debugging
        init_grid_file = open(os.path.join(self.model_location, 'initial_grid.txt'), 'w')
        for row_idx in range(self.r):
            printline = []
            for col_idx in range(self.c):
                if self.wf_grid[row_idx][col_idx] == None:
                    printline.append('<<{}>>'.format(self.wf_grid[row_idx][col_idx]))
                else:
                    printline.append(self.wf_grid[row_idx][col_idx])
            init_grid_file.write('{}\n'.format('\t'.join(printline)))
        init_grid_file.close()
        stderr.write('Finished writing out initial grid.\n')

        # 2. Write out all train and dev instances
        train_file, dev_file = open(extrinsic_inflection.train, 'w'), open(extrinsic_inflection.dev, 'w')
        for row_idx in range(self.r):
            row, wfs, col_idxs = self.get_row(row_idx)
            if len(col_idxs) > 1:
                for trg_col_idx in col_idxs:
                    trg_wf = row[trg_col_idx]
                    src_col_idxs = list(x for x in col_idxs if x != trg_col_idx)
                    for src_col_idx in src_col_idxs:
                        src_wf = row[src_col_idx]
                        instance = '<' + str(src_col_idx) + '>' + ' ' + ' '.join(list(src_wf)) + ' ' + '<' + str(trg_col_idx) + '>' + '\t' + ' '.join(list(trg_wf))
                        if random.choice(range(10)) == 4:
                            dev_file.write('{}\n'.format(instance))
                        else:
                            train_file.write('{}\n'.format(instance))
        train_file.close()
        dev_file.close()

        # 3. Run the model from scratch.. don't return anything
        extrinsic_inflection.patience = 12
        trained_model = seq2seq_runner.run(extrinsic_inflection)

        # 4. Read in Dev predictions and rank best source cells for each target cell
        error = False
        trg_2_src_acc = dict((trg_col_idx, dict((src_col_idx, [0, 1]) for src_col_idx in range(self.c) if src_col_idx != trg_col_idx)) for trg_col_idx in range(self.c))
        preds = os.path.join(ext_model_location, 'predictions_dev.txt')
        for line in open(preds):
            line = line.strip()
            if line.startswith('SRC: '):
                src_col_idx = int(line.split('<', 1)[1].split('>', 1)[0])
                trg_col_idx = int(line.split('<')[-1].split('>')[0])
                trg_2_src_acc[trg_col_idx][src_col_idx][1] += 1
                if not error:
                    trg_2_src_acc[trg_col_idx][src_col_idx][0] += 1

            if '*ERROR*' in line:
                error = True
            else:
                error = False
        for trg_col_idx in trg_2_src_acc:
            for src_col_idx in trg_2_src_acc[trg_col_idx]:
                trg_2_src_acc[trg_col_idx][src_col_idx] = trg_2_src_acc[trg_col_idx][src_col_idx][0] / trg_2_src_acc[trg_col_idx][src_col_idx][1]

        trg_2_best_srcs = dict((trg_col_idx, list(trg_2_src_acc[trg_col_idx])) for trg_col_idx in range(self.c))
        for trg_col_idx in trg_2_best_srcs:
            trg_2_best_srcs[trg_col_idx].sort(key = lambda x : trg_2_src_acc[trg_col_idx][x], reverse=True)
            stderr.write('Best Predictors for cell {}:\n'.format(trg_col_idx))
            for best_src in trg_2_best_srcs[trg_col_idx]:
                stderr.write('\t{} ({})\n'.format(best_src, trg_2_src_acc[trg_col_idx][best_src]))

        # 5. Write out test set trying to predict each unattested cell from its best available predictor
        ext_model_location = os.path.join(self.model_location, 'Extrinsic_grid_completion_final')
        if not os.path.isdir(self.model_location):
            os.makedirs(self.model_location)
        extrinsic_inflection = Inflector(ext_model_location, data_format=dataloader.DataFormat.MT, extrinsic=True)
        extrinsic_inflection.train = None
        extrinsic_inflection.dev = None
        extrinsic_inflection.test = os.path.join(self.model_location, 'ext_test.tsv')
        extrinsic_inflection.checkpoint_to_restore = trained_model.best_checkpoint_path

        empty_slots = []
        test_file = open(extrinsic_inflection.test, 'w')
        for row_idx in range(self.r):
            row, wfs, col_idxs = self.get_row(row_idx)
            if None in row and len(col_idxs) > 0:
                for trg_col_idx in range(self.c):
                    if trg_col_idx not in col_idxs:
                        # Make one prediction for every empty cell in grid
                        if self.baseline == 'random_src':
                            src_col_idx = random.choice(col_idxs)
                        else:
                            for src_col_idx in trg_2_best_srcs[trg_col_idx]:
                                if src_col_idx in col_idxs:
                                    break
                        src_wf = row[src_col_idx]
                        instance = '<' + str(src_col_idx) + '>' + ' ' + ' '.join(list(src_wf)) + ' ' + '<' + str(trg_col_idx) + '>' + '\tPredictMe'
                        test_file.write('{}\n'.format(instance))
                        empty_slots.append((row_idx, trg_col_idx))
        test_file.close()

        # 6. Continue training the model on dev for one epoch and make predictions on test
        _ = seq2seq_runner.run(extrinsic_inflection)

        # 7. Parse the predictions file
        full_grid = np.array(self.wf_grid)
        preds = os.path.join(ext_model_location, 'predictions_test.txt')
        for line in open(preds):
            line = line.strip()
            if line.startswith('PRD:'):
                row_idx, col_idx = empty_slots.pop(0)
                pred = ''.join(line.split(':', 1)[1].split()).replace('_', ' ')
                assert self.wf_grid[row_idx][col_idx] == None
                full_grid[row_idx][col_idx] = pred

        # 8. Write out the completed grid for debugging
        full_grid_file = open(os.path.join(ext_model_location, 'pred_full_grid.txt'), 'w')
        for row_idx in range(self.r):
            printline = []
            for col_idx in range(self.c):
                if self.wf_grid[row_idx][col_idx] == None:
                    printline.append('<<{}>>'.format(full_grid[row_idx][col_idx]))
                else:
                    printline.append(full_grid[row_idx][col_idx])
            full_grid_file.write('{}\n'.format('\t'.join(printline)))
        full_grid_file.close()

        if len(empty_slots) != 0:
            raise Exception('{}\n\nHow did test instances and predictions get misaligned!?\n\t{}\n\t{}'.format(str(full_grid), len(empty_slots), '\n\t'.join(list(str(x) for x in empty_slots))))

        return full_grid

    def reinitialize_with_weighted_scores(self, orig_col_idx_2_wf, orig_wf_2_col_idx):

        # Get column exponent probs and reset everything else
        self.joint_sort_rows()
        self.wf_grid, self.exp_grid = np.array([[None]*self.c]*self.r), np.array([[None]*self.c]*self.r)  # rXc grids
        self.row_bases = np.array([None]*self.r)
        self.first_empty_row = 0
        self.first_singleton = None
        self.col_2_wf_2_pot_par_mates = self.orig_col_2_wf_2_pot_par_mates
        # Assign initial grid coordinates to each wf
        first_row_to_consider = 0
        for min_col_idx in range(self.c):
            # Start new paradigms for all words not fitting a previous paradigm
            for wf in orig_col_idx_2_wf[min_col_idx]:
                if self.first_empty_row == None:
                    # Updates first_empty_row during assignment
                    self.add_paradigm()
                self.assign(wf, (self.first_empty_row, min_col_idx))
                del orig_wf_2_col_idx[wf]
            del orig_col_idx_2_wf[min_col_idx]
            # For each new singleton-so-far par, consider filling each of the cells to the right of min_col_idx with a yet unassigned form s.t. we maximize  |paradigm|*|lcs(paradigm)| - sum(|affix| for affix in exponent for exponent in exponents)
            for row_idx in range(first_row_to_consider, min(x for x in (self.r, self.first_empty_row) if x)):
                # Take the best option from the final sorted beam
                options, best_score, best_wfs, best_row = self.get_init_row(row_idx, min_col_idx, orig_col_idx_2_wf, use_probs=True)
                best_new_wf = None
                for col_idx in range(min_col_idx+1, self.c):
                    best_new_wf = best_row[col_idx]
                    if best_new_wf != None:
                        self.assign(best_new_wf, (row_idx, col_idx))
                        # Once assigned, remove from eligible dictionaries
                        orig_col_idx_2_wf[col_idx].remove(best_new_wf)
                        del orig_wf_2_col_idx[best_new_wf]
                        del self.col_2_wf_2_pot_par_mates[col_idx][best_new_wf]
                self.update_row_seg_analyses(row_idx)

                # Track progress
                if row_idx % 500 == 0:
                    stderr.write('\tinitialization progress: {}%\n'.format(round(100*(row_idx/self.r), 4)))
                    if self.debug:
                        stderr.write('\t\t**singleton and best-non-singleton candidate paradigms**\n') 
                        for ranked_tup in options:
                            stderr.write('\t\tScore: {} \t{}\n'.format(ranked_tup[0], ', '.join(str(x) for x in ranked_tup[2])))    
                    stderr.flush()

            # Update and sanity check
            first_row_to_consider = row_idx + 1

        assert len(orig_wf_2_col_idx) == len(orig_col_idx_2_wf) == 0

    def get_init_row(self, row_idx, min_col_idx, col_idx_2_wf, use_probs=False):
        # Initialize trivial beam search (legitimate beams didn't improve performance)
        row, _, col_idxs = self.get_row(row_idx)
        row, col_idxs = list(row), list(col_idxs)
        wf = self.wf_grid[row_idx][min_col_idx]
        min_base_score = len(wf)
        best_score = 0
        beam = [(min_base_score, [wf], row, col_idxs)]
        # For each non-initial column, increment a beam that maximizes the objective
        for col_idx in range(min_col_idx+1, self.c):
            next_beam = list(beam)
            # Limit search space with nearest neighbors
            for new_wf in self.col_2_wf_2_pot_par_mates[min_col_idx][wf][col_idx]:
                # Check score of adding a potential new wf to this par in this col
                if new_wf in col_idx_2_wf[col_idx]:
                    for candidate in beam:
                        new_wfs = candidate[1] + [new_wf]
                        new_col_idxs = candidate[3] + [col_idx]
                        if use_probs:
                            base_score = self.get_base_len_score(new_wfs, None, col_idxs=new_col_idxs)
                        else:
                            base_score = self.get_base_len_score(new_wfs, None)
                        if base_score > best_score:
                            best_score = base_score
                            new_row = list(candidate[2])
                            new_row[col_idx] = new_wf
                            next_beam = [beam[0], (best_score, new_wfs, new_row, new_col_idxs)]
            beam = next_beam
        del self.col_2_wf_2_pot_par_mates[min_col_idx][wf]
        # Return the best option in the final beam
        if best_score >= min_base_score:
            return beam, beam[1][0], beam[1][1], beam[1][2]
        else:
            return beam, beam[0][0], beam[0][1], beam[0][2]

    def init_col_clusters(self, wfs, masked_embeddings=True, target_affix_embeddings=True, target_syntactic_windows=True):
        # Get word embeddings with different inductive biases
        self.ft_syn_model, wf_matrix, vec_2_wf, _ = get_fasttext_vectors(wfs, self.sents, inductive_bias='syntactic', masked_embeddings=masked_embeddings, target_affix_embeddings=target_affix_embeddings, target_syntactic_windows=target_syntactic_windows)
        self.ft_sem_model, _, _, _ = get_fasttext_vectors(wfs, self.sents, inductive_bias='semantic')
        # Cluster forms into c cell columns
        initial_clusters, _, self.c = custom_cluster(wf_matrix, self.c)
        wf_2_col_idx, col_idx_2_wf = self.parse_clustering_output(initial_clusters, vec_2_wf)
        # Learn embedding neighborhoods specific to each column cluster
        self.get_column_matrices(col_idx_2_wf)  # limits pot par mates
        return wf_2_col_idx, col_idx_2_wf

    def parse_clustering_output(self, initial_clusters, vec_2_wf):
        # Get wf_2_cluster and cluster_2_wf dictionaries
        wf_2_cluster = {}  # Yet unassigned wfs which must be assigned
        cluster_2_wf = dict((cluster_idx, set()) for cluster_idx in range(len(initial_clusters)))

        # Prepare random baselines
        if self.baseline == 'random':
            self.r = len(self.gold.lem_2_wf)
            self.c = len(self.gold.cell_2_wf)
            self.wf_grid, self.exp_grid, self.row_bases = np.array([[None]*self.c]*self.r), np.array([[None]*self.c]*self.r), np.array([None]*self.r)
            slots = list(range(self.r*self.c))
            random.shuffle(slots)
        # Prepare singleton baseline
        elif self.baseline == 'all_singletons':
            self.r, cntr = len(vec_2_wf), 0
            self.wf_grid, self.exp_grid, self.row_bases = np.array([[None]*self.c]*self.r), np.array([[None]*self.c]*self.r), np.array([None]*self.r)

        # Actually parse the clusters
        for cluster_idx in range(len(initial_clusters)):
            for wf_idx in range(len(initial_clusters[cluster_idx])):
                cluster_vec = initial_clusters[cluster_idx][wf_idx]
                wf = vec_2_wf[tuple(list(cluster_vec))]
                wf_2_cluster[wf] = cluster_idx
                cluster_2_wf[cluster_idx].add(wf)

                # all_singletons baseline assumes one cell and one word per paradigm
                if self.baseline == 'all_singletons':
                    if cntr == self.r: self.add_paradigm()
                    self.assign(wf, (cntr, 0))
                    self.row_bases[cntr] = '{}_{}'.format(cntr, wf)
                    self.exp_grid[cntr][0] = ()
                    cntr += 1
                # random baseline assigns each wf to a random slot in the grid
                elif self.baseline == 'random':
                    row_idx, col_idx = divmod(slots.pop(0), self.c)
                    self.assign(wf, (row_idx, col_idx))

        # Evaluate singletons baseline
        if self.baseline == 'all_singletons':
            self.c = 1
            self.validate(msg='{} Grid evaluation one cell, singleton paradigm baseline'.format(self.lg))
            exit()
        # Evaluate random assignment baselines
        elif self.baseline == 'random':
            self.validate(msg='{} Grid evaluation of random assignment baseline'.format(self.lg))
            exit()

        return wf_2_cluster, cluster_2_wf

    def get_column_matrices(self, col_idx_2_wf):

        stderr.write('Getting vocabulary embedding neighborhoods..\n')
        stderr.flush()
        # Get neighborhoods by column to reduce search space
        col_idx_2_wf_matrix = dict((c, {}) for c in range(self.c))
        for col_idx in col_idx_2_wf:
            col_idx_2_wf_matrix[col_idx]['row_2_wf'] = list(col_idx_2_wf[col_idx])
            col_idx_2_wf_matrix[col_idx]['wf_2_row'] = dict((col_idx_2_wf_matrix[col_idx]['row_2_wf'][row_idx], row_idx) for row_idx in range(len(col_idx_2_wf_matrix[col_idx]['row_2_wf'])))
            # Extended vectors
            col_idx_2_wf_matrix[col_idx]['matrix'] = np.array(list(np.concatenate((self.ft_sem_model[wf], self.ft_syn_model[wf]), axis=0) for wf in col_idx_2_wf_matrix[col_idx]['wf_2_row']))
            # Learn nearest neighborhood
            col_idx_2_wf_matrix[col_idx]['neighborhood'] = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(col_idx_2_wf_matrix[col_idx]['matrix'])), algorithm='ball_tree').fit(col_idx_2_wf_matrix[col_idx]['matrix'])
        # Precompute all nearest neighbors for faster run time with less memory
        self.col_2_wf_2_pot_par_mates = {}
        self.orig_col_2_wf_2_pot_par_mates = {}
        for src_col_idx in col_idx_2_wf_matrix:
            self.col_2_wf_2_pot_par_mates[src_col_idx] = {}
            self.orig_col_2_wf_2_pot_par_mates[src_col_idx] = {}
            for wf in col_idx_2_wf_matrix[src_col_idx]['row_2_wf']:
                self.col_2_wf_2_pot_par_mates[src_col_idx][wf] = {}
                self.orig_col_2_wf_2_pot_par_mates[src_col_idx][wf] = {}
                # Get source vector
                wf_vector = col_idx_2_wf_matrix[src_col_idx]['matrix'][col_idx_2_wf_matrix[src_col_idx]['wf_2_row'][wf]].reshape(1, -1)
                for trg_col_idx in range(self.c):
                    if trg_col_idx != src_col_idx:
                        # Get neighbors conditional on target column
                        neighbor_idxs = col_idx_2_wf_matrix[trg_col_idx]['neighborhood'].kneighbors(wf_vector)[1][0]
                        self.col_2_wf_2_pot_par_mates[src_col_idx][wf][trg_col_idx] = list(col_idx_2_wf_matrix[trg_col_idx]['row_2_wf'][neighbor_idx] for neighbor_idx in neighbor_idxs)
                        self.orig_col_2_wf_2_pot_par_mates[src_col_idx][wf][trg_col_idx] = list(self.col_2_wf_2_pot_par_mates[src_col_idx][wf][trg_col_idx])
            # Track progress
            stderr.write('\tprogress: {}%\n'.format(round(100*(src_col_idx/self.c), 4)))
            stderr.flush()

    #############################################################
    ### Global Update Functions
    #############################################################
    def update_all_col_probs(self):
        """Get exponent likelihoods conditional on column membership.. Only count first 'stop_at' paradigms."""
        if self.debug:
            stderr.write('Exponent Probabilities by Column\n')
        # Initialize exponent data structures
        self.col_exp_probs = dict((col_idx, {}) for col_idx in range(self.c))
        self.col_ranked_exps = {}
        denoms = [0] * self.c
        # Count all exponent and affix occurrences conditional on column
        for row_idx in range(min(x for x in (self.first_singleton, self.first_empty_row, self.r) if x)):
            cntr = 0
            for col_idx in self.get_row(row_idx)[2]:
                cntr += 1
                denoms[col_idx] += 1
                exp = self.exp_grid[row_idx][col_idx]
                if exp == None:
                    self.update_row_seg_analyses(row_idx)
                    exp = self.exp_grid[row_idx][col_idx]
                if exp not in self.col_exp_probs[col_idx]:
                    self.col_exp_probs[col_idx][exp] = Averager()
                self.col_exp_probs[col_idx][exp].increment_numerator(1)
            assert cntr > 1
        # Normalize probabilities
        for col_idx in range(self.c):
            for exp in self.col_exp_probs[col_idx]:
                self.col_exp_probs[col_idx][exp].increment_denominator(denoms[col_idx])
                self.col_exp_probs[col_idx][exp].get_average()
            # Rank probabilities
            self.col_ranked_exps[col_idx] = list(x[1] for x in sorted(list((self.col_exp_probs[col_idx][exp].average, exp) for exp in self.col_exp_probs[col_idx]), reverse=True))
            if self.debug:
                stderr.write('\tColumn {}:\n\t\t{}\n'.format(col_idx, '\n\t\t'.join(str(exp) for exp in self.col_ranked_exps[col_idx][0:10])))

    def joint_sort_rows(self):
        """Jointly sorts wf_grid, exp_grid, and row_bases by descending row length (where None values do not count toward length)"""
        # Sort row_idxs
        row_idxs = list(range(self.r))
        row_idxs.sort(key = lambda x : len(self.get_row(x)[1]), reverse=True)
        # Initialize new grids
        wf_grid, exp_grid = np.array([[None]*self.c]*self.r), np.array([[None]*self.c]*self.r)
        row_bases = np.array([None]*self.r)
        # Populate new grids according to order of sorted row_idxs
        self.first_singleton, self.first_empty_row = None, None
        for trg_idx in range(self.r):
            src_idx = row_idxs[trg_idx]
            wf_grid[trg_idx] = self.wf_grid[src_idx]
            row_bases[trg_idx] = self.row_bases[src_idx]
            exp_grid[trg_idx] = self.exp_grid[src_idx]
            if self.first_singleton == None:
                if len(self.get_row(src_idx)[1]) == 1:
                    self.first_singleton = trg_idx
            elif self.first_empty_row == None:
                if len(self.get_row(src_idx)[1]) == 0:
                    self.first_empty_row = trg_idx
        # Overwrite old grids
        self.wf_grid, self.exp_grid, self.row_bases = wf_grid, exp_grid, row_bases
        # Sanity check
        if self.first_empty_row != None:
            assert self.first_empty_row == 0 or len(self.get_row(self.first_empty_row-1)[1]) > 0
            assert len(self.get_row(self.first_empty_row)[1]) == 0
        else:
            assert len(self.get_row(self.r-1)[1]) > 0
        # Update exponent probabilities
        self.update_all_col_probs()

    #############################################################
    ### Row/Column Update Functions
    #############################################################
    def add_paradigm(self):
        if self.first_empty_row == None:
            self.first_empty_row = self.r
        self.r += 1
        self.wf_grid = np.array(list(self.wf_grid) + [[None]*self.c])
        self.exp_grid = np.array(list(self.exp_grid) + [[None]*self.c])
        self.row_bases = np.array(list(self.row_bases) + [None])

    def update_row_seg_analyses(self, row_idx):
        """Updates base and exponent grid rows"""
        row, wfs, col_idxs = self.get_row(row_idx)
        if len(wfs) > 1:  # multi-form
            base = lcs(row)
            self.row_bases[row_idx] = '{}_{}'.format(row_idx, base)
            self.exp_grid[row_idx] = getExponent(base, row)
        elif len(wfs) == 1:  # singleton paradigms
            self.update_singleton_seg(wfs[0], row_idx, col_idxs[0])
        else:
            self.row_bases[row_idx] = None
            self.exp_grid[row_idx] = np.array([None]*self.c)

    #############################################################
    ### Local Update Functions
    #############################################################
    def update_singleton_seg(self, wf, row_idx, col_idx):
        base, exp = self.get_singleton_seg(wf, col_idx)
        # If no exponent was feasible, assume zero exponence
        if base == None:
            base, exp = wf, ()
            if exp not in self.col_exp_probs[col_idx]:
                # Update relevant structures to accomodate the zero exponent
                self.col_ranked_exps[col_idx].append(exp)
                self.col_exp_probs[col_idx][exp] = Averager()
                self.col_exp_probs[col_idx][exp].increment_numerator(self.min_exp_prob)
                self.col_exp_probs[col_idx][exp].increment_denominator(1+self.min_exp_prob)
                self.col_exp_probs[col_idx][exp].get_average()
        # Update row_bases and exp_grid accordingly
        self.row_bases[row_idx] = '{}_{}'.format(row_idx, base)
        self.exp_grid[row_idx][col_idx] = exp        

    def assign(self, wf, grid_coord, update_analyses=False):
        """Assign in wf_grid; update row_bases and exp_grid if update_analyses"""
        self.wf_grid[grid_coord[0]][grid_coord[1]] = wf
        # Check if we need to update where the empty rows start
        if self.first_empty_row == grid_coord[0] and wf != None:
            self.first_empty_row += 1
            if self.first_empty_row == self.r:
                self.first_empty_row = None
        # Update base and exponent analyses based on new assignment
        if update_analyses:
            self.update_row_seg_analyses(grid_coord[0])

    def swap(self, src_grid_coord, trg_grid_coord):
        (x, y) = src_grid_coord
        (xx, yy) = trg_grid_coord
        src_wf, trg_wf = self.wf_grid[x][y], self.wf_grid[xx][yy]
        # Swap assignments across wf_grid, row_bases, and exp_grid
        self.wf_grid[x][y] = trg_wf
        self.wf_grid[xx][yy] = src_wf
        for row_idx in (x, xx):
            self.update_row_seg_analyses(row_idx)

    #############################################################
    ### Utility Functions
    #############################################################
    def get_base_len_score(self, wfs, col_idx, col_idxs=None):
        """Intuitively, this returns the total number of characters across all word forms that participate in the analyzed base less the characters that do not"""
        if len(wfs) > 1:
            base = lcs(wfs)
            if self.exponent_penalty:
                if col_idxs == None:
                    return 2 * len(base) * len(wfs) - sum(len(wf) for wf in wfs)
                else:
                    exps = getExponent(base, wfs)
                    base_reward = len(base) * len(wfs)
                    exp_penalty = 0
                    for col_idx in col_idxs:
                        exp = exps.pop(0)
                        exp_penalty += self.get_penalty(exp, col_idx)
                    return base_reward - exp_penalty
            else:
                return len(base) * len(wfs)
        else:
            base, _ = self.get_singleton_seg(wfs[0], col_idx)
            if self.exponent_penalty:
                return 2 * len(base) - len(wfs[0])
            else:
                return len(base)

    # @lru_cache(maxsize=256)
    def get_penalty(self, exp, col_idx):
        penalty = 0
        for aff in exp:
            aff = aff.replace('<', '').replace('>', '')
            penalty += len(aff)
        if exp == self.col_ranked_exps[col_idx][0]:
            penalty = 0
        elif exp in self.col_ranked_exps[col_idx]:
            penalty *= (1 + (1 - (self.col_exp_probs[col_idx][exp].average / self.col_exp_probs[col_idx][self.col_ranked_exps[col_idx][0]].average)))
        else:
            penalty *= 2

        return penalty

    def get_singleton_seg(self, wf, col_idx):
        """ Find highest ranked feasible exponent, deduce corresponding base"""
        best_base, best_exp = None, None
        for exp in self.col_ranked_exps[col_idx]:
            form = '<{}>'.format(wf)
            base = ''
            match = True
            for a in exp:
                if a in form:
                    base += form.split(a, 1)[0]
                    form = form.split(a, 1)[1]
                else:
                    match = False
                    break
            if match:
                best_base = base + form[:-1]
                best_base = best_base.replace('<', '', 1)
                best_exp = exp
                break
        return best_base, best_exp

    def get_row(self, row_idx):
        row = self.wf_grid[row_idx]
        col_idxs = np.nonzero(row)[0]
        wfs = row[col_idxs]
        return row, wfs, col_idxs

    def get_column(self, col_idx):
        col = self.wf_grid.transpose()[col_idx]
        row_idxs = np.nonzero(col)[0]
        wfs = col[row_idxs]
        return col, wfs, row_idxs

    def debug_print_grid(self):
        stderr.write('\n____Sample Paradigms____\n')
        for row_idx in sorted(random.sample(list(range(self.r)), 10)):
            row, wfs, col_idxs = self.get_row(row_idx)
            if len(wfs) > 0:
                stderr.write('{}) {}\n'.format(row_idx, '\t'.join(list('{}->{}'.format(col_idxs[idx], wfs[idx]) for idx in range(len(wfs))))))
                if len(wfs) > 1:
                    base = lcs(row)
                    exps = getExponent(base, row)
                else:
                    base, exp = self.get_singleton_seg(wfs[0], col_idxs[0])
                    exps = np.array([None]*self.c)
                    exps[col_idxs[0]] = exp
                stderr.write('Paradigm {}: {}\n'.format(row_idx, row))
                stderr.write('\tBase: {}\n'.format(base))
                for col_idx in range(self.c):
                    if col_idx in col_idxs:
                        wf, exp = row[col_idx], exps[col_idx]

                        if exp in self.col_exp_probs[col_idx]:
                            exp_prob = self.col_exp_probs[col_idx][exp].average
                        else:
                            exp_prob = self.min_exp_prob
                        stderr.write('\tCell {}: {} \t--> \t{}\n'.format(col_idx, wf, exp))
                    else:
                        stderr.write('\tCell {}: <EMPTY>\n'.format(col_idx))

    def get_analyses(self):

        # Initialize data structures
        self.base_2_wf, self.wf_2_base = {}, {}
        self.col_idx_2_wf, self.wf_2_col_idx, self.base_2_IC = {}, {}, {}

        # Map each unique signature to IC label
        sigs = set()
        for row_idx in range(self.r):
            sigs.add(tuple(self.exp_grid[row_idx]))
        sigs = list(sigs)
        sig_2_IC = dict((sigs[sig_idx], sig_idx) for sig_idx in range(len(sigs)))

        # Extract cluster analyses from grids
        for row_idx in range(self.r):
            _, wfs, col_idxs = self.get_row(row_idx)
            # Make sure we don't record an empty row
            if len(wfs) > 0:
                signature = tuple(self.exp_grid[row_idx])
                base = self.row_bases[row_idx]

                # TODO: Exception handling when base already exists and it's not a baseline
                if base == None or base in self.base_2_wf:
                    if self.baseline == 'random':
                        if base == None:
                            base = 'q'
                        while base in self.base_2_wf:
                            base += 'q'
                    else:
                        raise Exception('Base {} is already in self.base_2_wf:\n\t{}\n\t{}\n\t{}'.format(base, str(self.base_2_wf[base]), str(wfs), str(col_idxs)))

                # Record information
                self.base_2_IC[base] = sig_2_IC[signature]
                self.base_2_wf[base] = set()
                for wf, col_idx in zip(wfs, col_idxs):
                    self.base_2_wf[base].add(wf)
                    if wf not in self.wf_2_base:
                        self.wf_2_base[wf] = set()
                    self.wf_2_base[wf].add(base)
                    if col_idx not in self.col_idx_2_wf:
                        self.col_idx_2_wf[col_idx] = set()
                    self.col_idx_2_wf[col_idx].add(wf)
                    if wf not in self.wf_2_col_idx:
                        self.wf_2_col_idx[wf] = set()
                    self.wf_2_col_idx[wf].add(col_idx)
           
    def validate(self, msg=None, eval_fn='intrinsic'):
        if eval_fn == 'extrinsic':
            self.extrinsic_accuracy, self.extrinsic_accuracy_partial = evaluate(self, eval_fn=eval_fn, msg=msg)
        else:
            self.cell_F, self.par_F, self.grid_F = evaluate(self, eval_fn=eval_fn, msg=msg)

