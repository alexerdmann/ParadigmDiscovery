from sys import stdout, stderr
from sklearn.metrics.cluster import v_measure_score
from math import log

def evaluate(grids, eval_fn='intrinsic', msg=None):

    if eval_fn == 'extrinsic':
        return eval_extrinsic(grids.ext_analogies, grids.full_grid, msg=msg)

    elif eval_fn == 'downstream':
        raise Exception('TODO: Code downstream evaluation using unsupervised analyses for MT.')

    elif eval_fn == 'intrinsic':
        return eval_intrinsic(grids, msg=msg)


def eval_intrinsic(grids, msg=None):

    # Track all evaluatable word forms
    wfs = set([])
    test_lem_2_wf, test_wf_2_lem, test_cell_2_wf, test_wf_2_cell = {}, {}, {}, {}
    # Only consider paradigms, cell, wf overlapping between UD and UM
    for lem in set(grids.gold.lem_2_cell_2_wf).intersection(set(grids.gold_UM_intersect.lem_2_cell_2_wf)):
        for cell in set(grids.gold.lem_2_cell_2_wf[lem]).intersection(set(grids.gold_UM_intersect.lem_2_cell_2_wf[lem])):
            for wf in grids.gold.lem_2_cell_2_wf[lem][cell].intersection(grids.gold_UM_intersect.lem_2_cell_2_wf[lem][cell]):

                # Found an intersecting lem,cell,wf realization
                # Now get all possible syncretisms within lem's paradigm
                for cell_sync in grids.gold_UM_intersect.wf_2_cell[wf].intersection(set(grids.gold.cell_2_wf)):
                    if wf in grids.gold_UM_intersect.lem_2_cell_2_wf[lem][cell_sync]:

                        # Update all data structures
                        if lem not in test_lem_2_wf:
                            test_lem_2_wf[lem] = set([])
                        if cell_sync not in test_cell_2_wf:
                            test_cell_2_wf[cell_sync] = set([])
                        if wf not in test_wf_2_cell:
                            test_wf_2_cell[wf] = set([])
                            test_wf_2_lem[wf] = set([])
                        wfs.add(wf)
                        test_lem_2_wf[lem].add(wf)
                        test_cell_2_wf[cell_sync].add(wf)
                        test_wf_2_cell[wf].add(cell_sync)
                        test_wf_2_lem[wf].add(lem)

    gold_cell_mates, gold_par_mates = get_mates(wfs, test_wf_2_lem, test_lem_2_wf, test_wf_2_cell, test_cell_2_wf)
    cell_mates, row_mates = get_mates(wfs, grids.wf_2_base, grids.base_2_wf, grids.wf_2_col_idx, grids.col_idx_2_wf)
    par_F, par_prec, par_rec, cell_F, cell_prec, cell_rec = get_cluster_scores(row_mates, cell_mates, gold_par_mates, gold_cell_mates)

    # Calculate Grid harmonic mean of cell and par F scores
    if par_F == 0 or cell_F == 0:
        grid_F = 0
    else:
        grid_F = 2 * ((par_F * cell_F) / (par_F + cell_F))

    # Get a readable results line
    par_F *= 100
    par_prec *= 100
    par_rec *= 100
    cell_F *= 100
    cell_prec *= 100
    cell_rec *= 100
    grid_F *= 100
    res_line = ','.join(list(str(round(x, 2)) for x in (len(grids.base_2_wf), grids.c, cell_F, cell_prec, cell_rec, par_F, par_prec, par_rec, grid_F)))

    # Write out
    if msg == None:
        write_to = stderr
    else:
        stdout.write('\n{}\n'.format(msg))
        write_to = stdout
    write_to.write('____________________________________\n')
    write_to.write('----Stats----\n')
    write_to.write('Language: {}\n'.format(grids.lg))
    write_to.write('Lexicon Forms: {}\n'.format(len(grids.wf_2_base)))
    write_to.write('Lexicon Paradigms: {}\n'.format(len(grids.gold.lem_2_wf)))
    write_to.write('Evaluated Paradigms: {}\n'.format(len(grids.gold_UM_intersect.lem_2_wf)))
    write_to.write('Cells: {}\n'.format(len(grids.gold_UM_intersect.cell_2_wf)))
    write_to.write('Proposed Paradigms: {}\n'.format(len(grids.base_2_wf)))
    write_to.write('Proposed Cells: {}\n'.format(grids.c))
    write_to.write('----Scores----\n')
    write_to.write('Cell F: {}\n'.format(round(cell_F, 2)))
    write_to.write('\tprec: {} \t rec: {}\n'.format(round(cell_prec, 2), round(cell_rec, 2)))
    write_to.write('Paradigm F: {}\n'.format(round(par_F, 2)))
    write_to.write('\tprec: {} \t rec: {}\n'.format(round(par_prec, 2), round(par_rec, 2)))
    write_to.write('Grid Harmony: {}\n'.format(round(grid_F, 2)))
    write_to.write('____________________________________\n')
    write_to.write('ResLine:\t{}\n'.format(res_line))
    stdout.flush()

    return cell_F, par_F, grid_F


def get_mates(wfs, wf_2_par, par_2_wf, wf_2_cell, cell_2_wf):
    # Get gold clustering evaluation data
    cell_mates, row_mates = dict((wf, set([])) for wf in wfs), dict((wf, set([])) for wf in wfs)
    for wf in wfs:
        for lem in wf_2_par[wf]:
            row_mates[wf].update(set(list(x for x in par_2_wf[lem] if x in wfs and x != wf)))
        for cell in wf_2_cell[wf]:
            cell_mates[wf].update(set(list(x for x in cell_2_wf[cell] if x in wfs and x != wf)))
    return cell_mates, row_mates


def get_cluster_scores(row_mates, cell_mates, gold_row_mates, gold_cell_mates):
    row_F, row_prec, row_rec = get_F(row_mates, gold_row_mates)
    cell_F, cell_prec, cell_rec = get_F(cell_mates, gold_cell_mates)
    return row_F, row_prec, row_rec, cell_F, cell_prec, cell_rec


def get_F(mates, gold_mates):
    correct, prec_denom, rec_denom = 0, 0.001, 0.001
    for wf in gold_mates:
        correct += len(mates[wf].intersection(gold_mates[wf]))
        prec_denom += len(mates[wf])
        rec_denom += len(gold_mates[wf])
    prec = correct / prec_denom
    rec = correct / rec_denom
    if prec == 0 or rec == 0:
        F = 0
    else:
        F = 2 * ((prec * rec) / (prec + rec))
    return F, prec, rec


def eval_extrinsic(analogies, full_grid, msg=None):
    """
    This doesn't explicitly check that there exists two distinct rows that satisfy the analogy, only that each possible pair of forms could appear in the same column/row at least once, independently of the other pairs. Functionally, this should behave nearly the same except that exploding the grid with syncretic forms s.t. each paradigm includes every word could trivially get 100% accuracy, though the intrinsic mutual information would penalize this harshly of course.
    """

    stderr.write('Evaluating extrinsically\n')

    # Check which rows and columns each word appears in
    proposed_pars, proposed_cells = 0, len(full_grid[0])
    wf_2_row, wf_2_col = {}, {}
    for row in range(len(full_grid)):
        got_one = False
        for col in range(len(full_grid[row])):
            wf = full_grid[row][col]
            if wf != None:
                if not got_one:
                    proposed_pars += 1
                    got_one = True
                if wf not in wf_2_row:
                    wf_2_row[wf] = set([])
                wf_2_row[wf].add(row)
                if wf not in wf_2_col:
                    wf_2_col[wf] = set([])
                wf_2_col[wf].add(col)

    # Initialize stats
    anal_num, anal_den = 0, 0
    expand_num, expand_den = 0, 0

    # Iterate through test instances
    for wf11, wf12, wf21, wf22 in analogies:
        anal_den += 1
        expand_den += 1
        stderr.write('ANALOGY: {} -> {} : {} -> {}\n'.format(wf11, wf12, wf21, wf22))

        # 0. Did we realize that wf22 is even a word?
        if wf22 in wf_2_row:
            expand_num += 1
            stderr.write('\twe predicted {} somewhere\n'.format(wf22))

            if wf21 in wf_2_row and wf12 in wf_2_row:
                row2s = wf_2_row[wf21].intersection(wf_2_row[wf22])
                col2s = wf_2_col[wf12].intersection(wf_2_col[wf22])
            
                # 1. Does it share a row with wf21?
                if len(row2s) > 0:
                    stderr.write('\tshares row with {}\n'.format(wf21))
                # 2. Does it share a column with wf12?
                if len(col2s) > 0:
                    stderr.write('\tshares column with {}\n'.format(wf12))
                # 3. Is the clustering analogical, i.e., Does wf11 complete the square?
                if len(row2s) > 0 and len(col2s) > 0:
                    if wf11 in wf_2_row:
                        row1s = wf_2_row[wf11].intersection(wf_2_row[wf12])
                        col1s = wf_2_col[wf11].intersection(wf_2_col[wf21])
                        if len(row1s) > 0 and len(col1s) > 0:
                            anal_num += 1
                            stderr.write('\tSUCCESSFULL ANALOGY (completed by {})\n'.format(wf11))

    anal_acc, expand_acc = 100*anal_num/anal_den, 100*expand_num/expand_den

    if msg == None:
        write_to = stderr
    else:
        stdout.write('\n{}\n'.format(msg))
        write_to = stdout
    write_to.write('____________________________________\n')
    write_to.write('----Stats----\n')
    write_to.write('Total Analogies Tested: {}\n'.format(anal_den))
    write_to.write('Proposed Paradigms: {}\n'.format(proposed_pars))
    write_to.write('Proposed Cells: {}\n'.format(proposed_cells))
    write_to.write('----Scores----\n')
    write_to.write('Analogy Accuracy: {}%\n'.format(round(anal_acc, 2)))
    write_to.write('Lexicon Expansion Accuracy: {}%\n'.format(round(expand_acc, 2)))
    write_to.write('____________________________________\n')
    write_to.write('ResLine:\t{},{}\n'.format(round(anal_acc, 2), round(expand_acc, 2)))

    return anal_acc, expand_acc
