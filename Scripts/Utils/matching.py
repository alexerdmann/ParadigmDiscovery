from sys import argv, exit
import numpy as np
from abc import ABC
from Utils.munkres import Munkres

"""
This code performs asymetrical matching of a dynamically weighted complete bipartite graph.
All weights are assumed to be positive to start. Mappings can feature many-inputs-to-one-output
relationships or one-input-to-many-outputs relationships, the only restriction is that all 
input nodes must be connected to at least one output node. The supported algorithms are the 
de facto, Hungarian/Munkres algorithm--which does not consider dynamic weights but is 
modified slightly to gauruntee that all input nodes connect--and the new Kilimanjaro 
algorithm, which incorporates tuneable parameters to govern the extent to which many-to-one 
or one-to-many mappings are disprefered as a function of the number of edges already assigned 
to either node. This is the sense in which the graph is "dynamically" weighted.

The design is meant to model natural language's dispreference for syncretism and over-abundance 
in the mappings of related word forms to their cells in morphological paradigms. Syncretism is 
one form mapping to multiple paradigm cells, e.g., in the HIT paradigm, 'hit' is both past 
tense and present, thus occupying 2 cells. Over-abundance is when one cell is occupied by 
multiple forms. There is usually a stronger dispreference for this than for syncretism, but it 
does occur, e.g., for many speakers, 'dived' and 'dove' are both perfectly acceptable forms in 
the past tense cell of the DIVE paradigm.

When the Kilimanjaro mapping algorithm selects an edge from the graph, it re-weights all other 
edges connecting to the same output cell node with an anti-over-abundance penalty. It also 
re-weights all other edges connecting to the same input word form node with an anti-syncretism 
penalty. The penalties are constants multiplied by the original row/column local maximum values 
of any edge connecting to that output cell node (over-abundance penalty) or connecting to that 
input form node (syncretism penalty).. original in that they refer to the edge weights in the 
original graph before the first penalty had been applied. This has the desirable property that 
when the penalty parameters are both set to 1.0 for a square matrix, Kilimanjaro approximates 
the Hungarian algorithm. Smaller syncretism/row and over-abundance/column penalties are 
progressively more permissive of the relevant phenomena. If they are set to zero however, the 
completely connected graph will be returned.

The algorithm stops running when all forms have been assigned to a cell node and the maximum 
weight of all remaining edges is less than 0. When both penalties are set to 1, the nXm matrix 
where n<m can exhibit syncretism, but is not required to. This is desirable as the paradigm may 
only be partially attested. With penalties both set to 1 however, over-abundance is prohibited 
unless m>n, in which case it is mandatory. This is also desirable given the assumption that 
every word form must realize some morphological cell. Tuning the penalties on real data with 
some independent objective function can provide insight to the correlation between people's 
dispreferences for syncretism and over-abundance. 
"""

####################################################################################
### Classes
####################################################################################

class bipartite_graph_super(ABC):

    def __init__(self, matrix, row_penalty=1.0,
        column_penalty=1.0, edge_type='weight'):

        ### Make a copy of the input matrix represented with numpy
        self.original = matrix
        self.weights = np.array(matrix)
        ### Initialize assignments
        self.assignments = {}
        ### Initialize parameters
        self.row_penalty = row_penalty
        self.column_penalty = column_penalty
        self.edge_type = edge_type

    def get_score(self, debug=False):

        total_dynamic_score = 0.0
        for form in self.assignments:
            for cluster in self.assignments[form]:
                if debug:
                    print('Alignment: {} \t Original Weight: {} \t Recorded Weight: {}'.format(str((form, cluster)), self.original[form][cluster], self.assignments[form][cluster]))
                total_dynamic_score += self.assignments[form][cluster]

        return total_dynamic_score


### Iterative matching selects highest cells over second highest in row or column, re-weights
    ## The difference between Kilimanjaro and Africa's second highest peak
    ## is greater than the comparable difference for any other continent
class bipartite_graph_Kilimanjaro(bipartite_graph_super):

    def handle_cost_weights(self):

        if self.edge_type == 'cost':
            self.costs = np.array(self.weights)
            max_weight = np.amax(self.weights)
            self.costs -= max_weight
            self.costs *= -1
            self.costs_trans = self.costs.transpose()

        self.weights_trans = self.weights.transpose()
        self.original_trans = np.array(self.original).transpose()

    def dynamic_asymmetric_matching(self):

        ### Initialization
        weights_viable = {}
        r_ranked = {} # [row][(weight, column),...ranked]
        c_ranked = {} # [column][(weight, row),...ranked]
        r_penalties = {} # horizontal against syncretism
        c_penalties = {} # vertical against over-abundance
        r_ranked, c_ranked, max_weight = self.get_rankings()
        for r in range(len(self.weights)):
            weights_viable[r] = {}
            if type(self.row_penalty) in (list, np.ndarray):
                r_penalties[r] = np.array(self.row_penalty)*r_ranked[r][0][0]
            else:
                r_penalties[r] = self.row_penalty*r_ranked[r][0][0] # penalizing syncretism
            for c in range(len(self.weights_trans)):
                weights_viable[r][c] = True
                c_penalties[c] = self.column_penalty*c_ranked[c][0][0] # penalizing over-abundance
        self.death_penalty = np.inf

        ### Iteratively assign an edge and re-weight the graph
        while len(self.assignments) < len(self.weights) or max_weight > 0.0:

            ### STEP 1: Greedily select the best edge to remove from the graph
                ## This equates to nullifying the best edge's cell in the matrix
            r_ranked, c_ranked, max_weight, best_R, best_C, weights_viable = self.assign_edge(weights_viable, r_ranked, c_ranked, max_weight, r_penalties, c_penalties)

            ### STEP 2: Re-weight the remaining edges
            ## STEP 2a: Reduce the likelihood of syncretism
                # Penalize best cell's row weights with row-specific penalty
                # Penalize best cell's column weights with column-specific penality
            r_ranked, c_ranked, max_weight = self.penalize_cross(best_R, best_C, r_penalties, c_penalties)

    def assign_edge(self, weights_viable, r_ranked, c_ranked, max_weight, r_penalties, c_penalties):

        ## Identify the highest weighted edge
        assigned_weight, best_R, best_C = self.select_edge(r_ranked, c_ranked, max_weight, r_penalties, c_penalties)
        r_ranked, c_ranked, max_weight = self.nullify_edge(r_ranked, c_ranked, best_R, best_C)

        ## Make sure the highest weighted edge has not already been chosen
        while weights_viable[best_R][best_C] == False:                
            r_ranked, c_ranked, max_weight = self.penalize_edge(r_ranked, c_ranked, best_R, best_C, self.death_penalty, max_weight)
            assigned_weight, best_R, best_C = self.select_edge(r_ranked, c_ranked, max_weight, r_penalties, c_penalties)

        ## Only permit peaks below sea level if the input node has not yet been assigned
        while assigned_weight <= 0.0 and best_R in self.assignments:
            r_ranked, c_ranked, max_weight = self.penalize_edge(r_ranked, c_ranked, best_R, best_C, self.death_penalty, max_weight)
            assigned_weight, best_R, best_C = self.select_edge(r_ranked, c_ranked, max_weight, r_penalties, c_penalties)

        ## Assign the match defined by that edge
        weights_viable[best_R][best_C] = False
        if best_R not in self.assignments:
            self.assignments[best_R] = {}
        self.assignments[best_R][best_C] = assigned_weight

        return r_ranked, c_ranked, max_weight, best_R, best_C, weights_viable

    def select_edge(self, r_ranked, c_ranked, max_weight, r_penalties, c_penalties):


        ## From all edges representing local maxima in either their row or column
            ## Select edge with greatest 1D prominence over second closest edge in its row or column
        max_prominence = None
        used = {}
        # for row maxima edges
        for r in r_ranked:
            height, c = r_ranked[r][0]
            # cache row maximum edge
            used[(r, c)] = True
            # prominence is the height minus what you're missing out on
                # by not climbing the next highest peak

            if type(r_penalties[r]) == np.ndarray:

                prominence = height - min(max(list(min(r_penalties[r][c2], max(0, self.weights[r][c2])) for c2 in c_penalties if c2 != c)), min(c_penalties[c], max(0, c_ranked[c][1][0]))) # max prominence in row or column

            else:
                prominence = height - min(min(r_penalties[r], max(0, r_ranked[r][1][0])), min(c_penalties[c], max(0, c_ranked[c][1][0]))) # max prominence in row or column
            if max_prominence == None or prominence > max_prominence:
                max_prominence = prominence
                best_R = r
                best_C = c
        # for column maxima edges
        for c in c_ranked:
            height, r = c_ranked[c][0]
            # that haven't already been cached
            if (r, c) not in used:

                if type(r_penalties[r]) == np.ndarray:

                    prominence = height - min(min(c_penalties[c], max(0, c_ranked[c][1][0])),
                        max(list(min(r_penalties[r][c2], max(0, self.weights[r][c2])) for c2 in c_penalties if c2 != c)))

                else:

                    prominence = height - min(min(c_penalties[c], max(0, c_ranked[c][1][0])), min(r_penalties[r], max(0, r_ranked[r][1][0]))) # max prominence in row or column

                if prominence > max_prominence:
                    max_prominence = prominence
                    best_R = r
                    best_C = c

        # print(self.weights)
        # print((best_R, best_C))
        # print(max_prominence)
        # print('\n###\n')

        return self.weights[best_R][best_C], best_R, best_C

    def nullify_edge(self, r_ranked, c_ranked, r, c):

        self.weights[r][c] = 0.0
        r_ranked[r] = list((self.weights[r][c], c) for c in range(len(self.weights[r])))
        r_ranked[r].sort(reverse=True)
        c_ranked[c] = list((self.weights_trans[c][r], r) for r in range(len(self.weights_trans[c])))
        c_ranked[c].sort(reverse=True)

        max_weight = max(list(r_ranked[r][0][0] for r in r_ranked))

        return r_ranked, c_ranked, max_weight

    def penalize_edge(self, r_ranked, c_ranked, r, c, penalty, max_weight):

        recalculate_max = False
        if self.weights[r][c] == max_weight:
            recalculate_max = True
        self.weights[r][c] -= penalty
        r_ranked[r] = list((self.weights[r][c], c) for c in range(len(self.weights[r])))
        r_ranked[r].sort(reverse=True)
        c_ranked[c] = list((self.weights_trans[c][r], r) for r in range(len(self.weights_trans[c])))
        c_ranked[c].sort(reverse=True)
        if recalculate_max:
            max_weight = max(list(r_ranked[r][0][0] for r in r_ranked))

        return r_ranked, c_ranked, max_weight

    def penalize_cross(self, best_R, best_C, r_penalties, c_penalties):

        ### Punish future syncretism


        if type(r_penalties[best_R]) == np.ndarray:
            self.weights[best_R] -= r_penalties[best_R]
        else:
            for c in range(len(self.weights[best_R])):
                self.weights[best_R][c] -= r_penalties[best_R]

        ### Punish future over-abundance
        for r in range(len(self.weights_trans[best_C])):
            self.weights[r][best_C] -= c_penalties[best_C]

        return self.get_rankings()

    def get_rankings(self):

        r_ranked = {}
        c_ranked = {}

        for r in range(len(self.weights)):
            r_ranked[r] = list((self.weights[r][c], c) for c in range(len(self.weights[r])))
            r_ranked[r].sort(reverse=True)
            # if type(self.row_penalty) in (np.ndarray, list):
            #     rest = []
            #     for l in r_ranked[r][1:]:
            #         weight = l[0]
            #         c = l[1]
            #         rest.append((min(r_penalties[r][c], max(0, weight)), c))


            #         min(r_penalties[r], max(0, r_ranked[r][1][0])), min(c_penalties[c], max(0, c_ranked[c][1][0]))
                # r_ranked[r] = [r_ranked[r][0]]
                # r_ranked[r].extend(rest)

            for c in range(len(self.weights_trans)):
                c_ranked[c] = list((self.weights_trans[c][r], r) for r in range(len(self.weights_trans[c])))
                c_ranked[c].sort(reverse=True)
        max_weight = max(list(r_ranked[r][0][0] for r in r_ranked)) 

        return r_ranked, c_ranked, max_weight 

    def handle_single_row_matrix(self):

        max_weight = np.amax(self.weights)
        l = list(self.weights[0])
        c = l.index(max_weight)
        self.assignments = {0: {c: max_weight}}

        if type(self.row_penalty) in (list, np.ndarray):
            penalty = np.array(self.row_penalty)*max_weight
        else:
            penalty = self.row_penalty * max_weight

        while max_weight > 0.0:
            self.weights[0][c] = 0.0
            self.weights -= penalty
            max_weight = np.amax(self.weights)
            if max_weight <= 0.0:
                break
            else:
                l = list(self.weights[0])
                c = l.index(max_weight)
                if c not in self.assignments[0]:
                    self.assignments[0][c] = max_weight


class bipartite_graph_Hungarian(bipartite_graph_super):

    def handle_cost_weights(self):

        max_weight = np.amax(self.weights)
        if self.edge_type == 'cost':
            self.costs = np.array(self.weights)
            self.weights -= max_weight
            self.weights *= -1
        else:
            self.costs = np.array(self.weights)
            self.costs -= max_weight
            self.costs *= -1

        self.weights_trans = self.weights.transpose()
        self.costs_trans = self.costs.transpose()

    def resize_matrix(self):

        row_count = self.original_row_count
        column_count = self.original_column_count

        if row_count != column_count:
            new_cost_matrix = list(list(row) for row in self.costs)

            while row_count != column_count:

                while row_count < column_count:
                    for row in self.costs:
                        row_count += 1
                        new_cost_matrix.append([0.0]*len(new_cost_matrix[0]))
                        

                while column_count < row_count:
                    for column in self.costs_trans:
                        column_count += 1
                        for r in range(len(new_cost_matrix)):
                            new_cost_matrix[r].append(0.0)


            self.costs = np.array(new_cost_matrix)
            self.costs_trans = self.costs.transpose()


        assert len(self.costs) == len(self.costs_trans)

    def asymetric_matching(self):

        ## initialize Hungarian algorithm, data structures, and penalties
        self.original = np.array(self.weights)
        self.original_row_count = len(self.weights)
        self.original_column_count = len(self.weights[0])
        self.r_penalties = list(max(row) for row in self.weights)
        self.c_penalties = list(max(column) for column in self.weights_trans)
        # self.resize_matrix()
        m = Munkres()
        self.assignments = {}
        
        ## assign mapping
        indexes, manipulated_matrix = m.compute(self.costs)
        for r, c in indexes:
            ## mappings that are not syncretic or over-abundant
            if r < self.original_row_count and c < self.original_column_count:
                w = self.weights[r][c]
                true_r = r
                true_c = c
            ## mappings that are syncretic or over-abundant shall be punished accordingly
                # unlike Kilimanjaro, Hungarian was blind to these penalties during mapping
            else:
                true_r = r%self.original_row_count
                factor_r = int(r/self.original_row_count)
                true_c = c%self.original_column_count
                factor_c = int(c/self.original_column_count)
                w = self.weights[true_r][true_c] - self.row_penalty*factor_r*self.r_penalties[true_r] - self.column_penalty*factor_c*self.c_penalties[true_c]

            ## Only record negative weight matches if all inputs have not yet been assigned 
            if len(self.assignments) < len(self.original) or w > 0.0:
                if true_r not in self.assignments:
                    self.assignments[true_r] = {}
                self.assignments[true_r][true_c] = w

    def handle_single_row_matrix(self):

        max_weight = np.amax(self.weights)
        l = list(self.weights[0])
        c = l.index(max_weight)
        self.assignments = {0: {c: max_weight}}


####################################################################################
### Functions
####################################################################################

def bipartite_match(matrix, row_penalty=1.0, column_penalty=1.0, assign_method='K2', edge_type='weight'):

    assert edge_type in ('weight', 'cost')
    if assign_method.lower() == 'kilimanjaro':
        G = bipartite_graph_Kilimanjaro(matrix, row_penalty=row_penalty, column_penalty=column_penalty, edge_type=edge_type)
        ## make sure graph edges reflect weights
        G.handle_cost_weights()
        ## learn mapping
        if len(G.weights) > 1: # standard, well behaved matrix
            G.dynamic_asymmetric_matching()
        else: # single row matrix
            G.handle_single_row_matrix()
        
        return G

    elif assign_method.lower() == 'hungarian':
        G = bipartite_graph_Hungarian(matrix, row_penalty=row_penalty, column_penalty=column_penalty)
        ## make sure graph edges reflect weights
        G.handle_cost_weights()
        ## learn mapping
        if len(G.weights) > 1: # standard, well behaved matrix
            G.asymetric_matching()
        else: # single row matrix
            G.handle_single_row_matrix()

        return G

    else:
        print('The bipartite matching algorithm {} is not supported.'.format(assign_method))
        exit()


####################################################################################
### Main
####################################################################################

if __name__ == '__main__':

    assign_method = argv[1]

    try:
        [syncretism_penalty, over_abundance_penalty] = [float(x) for x in argv[-2:]]
    except:
        syncretism_penalty = 0.5 #[1.0, 0.8, 0.8, 1.0]
        over_abundance_penalty = 1.0

    ##################################################

    # print('\nPARTIAL PARADIGM')
    # mat = np.array([
    #     [1., 12., 3.,  4.],
    #     [2., 14., 6.,  8.],
    #     [3., 12., 9.,  6.]
    #     ])
    # best_score = 29 # 12, 8, 9

    # print(mat)
    # G = bipartite_match(mat, row_penalty=syncretism_penalty, column_penalty=over_abundance_penalty, assign_method=assign_method)
    # score = G.get_score(debug=True)
    # print('Dynamic Score: {}\nBest Possible Score: {}\n'.format(float(score), float(best_score)))

    ##################################################

    print('\nSYNCRETIC PARADIGM')
    mat = np.array([
        [1., 12., 3.,  4.],
        [2., 46., 46.,  8.],
        [3., 12., 9.,  7.]
        ])
    best_score = 65 # 12, 46, 6

    print(mat)
    G = bipartite_match(mat, row_penalty=syncretism_penalty, column_penalty=over_abundance_penalty, assign_method=assign_method)
    score = G.get_score(debug=True)
    print('Dynamic Score: {}\nBest Possible Score: {}\n'.format(float(score), float(best_score)))


    ###################################################

    # print('\nOVER-ABUNDANT PARADIGM')
    # mat = np.array([
    #     [1., 2., 3.],
    #     [2., 4., 6.],
    #     [3., 6., 9.],
    #     [4., 7., 10.]
    #     ])
    # best_score = 15 # 1(-3), 2, 6, 10

    # print(mat)
    # G = bipartite_match(mat, row_penalty=syncretism_penalty, column_penalty=over_abundance_penalty, assign_method=assign_method)
    # score = G.get_score(debug=True)
    # print('Dynamic Score: {}\nBest Possible Score: {}\n'.format(float(score), float(best_score)))


    # ##################################################

    # print('\nPROBLEM FOR GREEDY EVEREST METHOD')
    # mat = np.array([
    #     [2., 200., 6.],
    #     [3., 200., 9.],
    #     [200., 201., 200.]
    #     ])
    # best_score = 409 # 200, 9, 200

    # print(mat)
    # G = bipartite_match(mat, row_penalty=syncretism_penalty, column_penalty=over_abundance_penalty, assign_method=assign_method)
    # score = G.get_score(debug=True)
    # print('Dynamic Score: {}\nBest Possible Score: {}\n'.format(float(score), float(best_score)))

    # ##################################################

    # print('\nEXTREMELY SYNCRETIC PARADIGM')
    # mat = np.array([
    #     [1., 12., 3.,  4., 3., 8., 2., 45],
    #     [2., 76., 76., 76.,  8., 11., 5., 9.],
    #     [3., 12., 9.,  6., 2., 31., 2., 2.]
    #     ])

    # print(mat)
    # G = bipartite_match(mat, row_penalty=syncretism_penalty, column_penalty=over_abundance_penalty, assign_method=assign_method)
    # score = G.get_score(debug=True)
    # print('Dynamic Score: {}\n'.format(float(score)))


    # ###################################################

    # print('\nEXTREMELY OVER-ABUNDANT PARADIGM')
    # mat = np.array([
    #     [1., 2., 3.],
    #     [2., 4., 6.],
    #     [3., 6., 29.],
    #     [2., 7., 42.],
    #     [7., 3., 3.],
    #     [9., 67., 19.],
    #     [1., 7., 7.],
    #     [4., 7., 10.]
    #     ])

    # print(mat)
    # G = bipartite_match(mat, row_penalty=syncretism_penalty, column_penalty=over_abundance_penalty, assign_method=assign_method)
    # score = G.get_score(debug=True)
    # print('Dynamic Score: {}\n'.format(float(score)))



