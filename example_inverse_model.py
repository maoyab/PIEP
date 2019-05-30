from param_sm_pdf import Processor
from datetime import datetime
from cPickle import dump, load
from os import path
#from mpi4py import MPI
import sys


# Some global parameters for the MH-MCMC that can be define
num_pl = 3                         # number of threads
samples = 20000                    # number of sample simulations per thread
max_num_pl = 6                     # max number of total run samples possible
n_pl_min = num_pl                  # min number of comnverged run samples to provide results
burnin = 1. / 2.                   # Fraction of burmn in period
efficiency_lim = [1, 95]           # limits of MH-MCMC acceptable efficiency

def loop(iterable_p):
    processor = Processor(model_params_estimate, model_type=model_type)
    for [n, loc_res] in iterable_p:
        dt0 = datetime.now()
        outfile_name = '%s_ix_%s_n.pickle' % (loc_res['loc_i'], n)
        theta0 = {'s_wilt': loc_res['s_wilt'],
                  's_star': loc_res['s_star'],
                  'delta': 0,
                  'rf_lambda': loc_res['rf_lambda'],
                  'rf_alpha': loc_res['rf_alpha'],
                  'Zr': loc_res['Zr'],
                  'b': loc_res['b'],
                  'Ks': loc_res['Ks'],
                  'n': loc_res['n'],
                  's_h': loc_res['s_h'],
                  's_fc': loc_res['s_fc'],
                  'f_w': loc_res['f_w'],
                  'f_max': loc_res['f_max'],
                  'et0': loc_res['et0'],
                  'et0_dry': loc_res['et0_dry'],
                  't_d': loc_res['t_d']
                 }
        p_ranges = {'s_wilt': [loc_res['s_h'], loc_res['s_fc']],
                    's_star': [loc_res['s_h'], loc_res['s_fc']],
                    'f_w': [0., 0.1],
                    'f_max': [0.1, 1],
                    }
        try:  
            pl_results, n_it, fail_conv_count, fail_eff_count = processor.get_mcmc_mh_results(loc_res['s_obs'],
                                                                          theta0, p_ranges,
                                                                          nbr_sim=samples, num_pl=num_pl,
                                                                          burnin=burnin, max_num_pl=max_num_pl,
                                                                          efficiency_lim=efficiency_lim)
            loc_res['n_it'] = n_it
            loc_res['num_pl'] = len(pl_results)
            loc_res['fail_conv_count'] = fail_conv_count
            loc_res['fail_eff_count'] = fail_eff_count

            if loc_res['num_pl'] == n_pl_min:
                loc_res = processor.process_raw_results(loc_res, pl_results, outfile_format=outfile_format)
            else:
                outfile_name = 'x_' + outfile_name

            picklename = path.join(resultpath, outfile_name)
            loc_res['ctime'] = (datetime.now() - dt0).seconds / 60.
            loc_res['fail_conv_count'] = fail_conv_count + loc_res['fail_conv_count']
            loc_res['fail_eff_count'] = fail_eff_count + loc_res['fail_eff_count']

            with open(picklename, 'wb') as f:
                dump(loc_res, f)
            print(loc_res['ctime'], loc_res['n_it'], loc_res['fail_conv_count'], loc_res['fail_eff_count'], outfile_name)
        except:
            print loc_res['loc_i'], n, '..............................................................'
            print theta0


def parallel():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    numit = len(iterable) / size
    diff = len(iterable) - numit * size
    if diff > 0:
        diff_list = [it for it in iterable[numit * size:]]
    iterable_p = [it for it in iterable[rank * numit:(rank + 1) * numit]]
    if rank < diff:
        iterable_p.append(diff_list[rank])
    loop(iterable_p)


if __name__ == '__main__':
    
    # example: python 'example_inverse_model.py' 'iterables/p_vmi_a_2012.pickle' 'full' 'A' 'e_max,s_wilt,s_star' 'results'

    outfile_format = sys.argv[2]                    # short or full to includes values of all all simulation estimates and original observations
    model_params_estimate = sys.argv[4].split(',')  # use names consistent with script dict such as 'e_max,s_wilt,s_star'
    model_type = sys.argv[3]                        # A for annual 
    filename = sys.argv[1]                          # iterable of dictionnaries to process
    resultpath = sys.argv[5]                        

    iterable = load(filename)
    loop(iterable)                                  # if processing in parallel with mpi4py use parallel() instead of loop()
    