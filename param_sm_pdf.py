import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp, percentileofscore
from sswm import SM_L, SM_D, SM_A


class Inverse_bayesian_fitting(object):
    def __init__(self, s_obs, unknown_params, p_ranges, model_type='A', nbr_sim=20000, burnin=1. / 2.):
        self.s_obs = s_obs
        self.burnin = burnin
        np.random.seed()
        self.nbr_sim = nbr_sim
        self.model_type = model_type
        if model_type == 'L':
            self.SM_PDF = SM_L
            self.full_param_name_list = ['s_h', 's_wilt', 's_star', 's_fc',
                                    'e_max', 'f_w',
                                    'rf_alpha', 'rf_lambda', 'delta',
                                    'a', 'b', 'Ks', 'n', 'Zr']
        elif model_type == 'D':
            self.SM_PDF = SM_D
            self.full_param_name_list = ['s_h', 's_wilt', 's_star', 's_fc',
                                    'e_max', 'f_w',
                                    'rf_alpha', 'rf_lambda', 'delta',
                                    'a', 'b', 'Ks', 'n', 'Zr', 'e_max_dry', 't_d']
        elif model_type == 'A':
            self.SM_PDF = SM_A
            self.full_param_name_list = ['s_h', 's_wilt', 's_star', 's_fc',
                                    'e_max', 'f_w',
                                    'rf_alpha', 'rf_lambda', 'delta',
                                    'a', 'b', 'Ks', 'n', 'Zr', 'e_max_dry', 't_d']

        self.unknown_params = unknown_params
        self.p_ranges = p_ranges

    def mcmc_mh(self, theta0_dict):
        accepted = 0.
        (li, theta_i) = self.init_random_model(theta0_dict)
        if (li, theta_i) != (np.nan, np.nan):
            result = [None] * (np.int(self.nbr_sim * (1 - self.burnin)))
            for it in range(self.nbr_sim):
                (acc, l_acc, theta_acc) = self.mcmc_mh_criteria(li, theta_i)
                if it >= self.nbr_sim * self.burnin:
                    theta_list = [getattr(theta_acc, vv) for vv in self.unknown_params]
                    result[np.int(it - self.nbr_sim * self.burnin)] = [l_acc, theta_list]
                    accepted = accepted + acc
                li, theta_i = l_acc, theta_acc
            return result, accepted / (self.nbr_sim * (1 - self.burnin)) * 100.
        else:
            return [np.nan], 'max init random'

    def init_random_model(self, params0_dict, maxcount=25):
        params = self.make_random_model(params0_dict)
        while_count = 0
        while self.test_model_consistency(params) < 1 and while_count < 1000:
            params = self.make_random_model(params0_dict)
            while_count = while_count + 1
        if while_count < 10:
            smpdf = self.SM_PDF(params)
            p0 = smpdf.get_p0()
            llk = self.eval_loglikelihood(p0, self.s_obs, params)
            if llk == 0 or np.isnan(llk):
                llk = -np.inf
            while_count = 0
            while llk == - np.inf and while_count < maxcount:
                while_count = while_count + 1
                params = self.make_random_model(params0_dict)
                while self.test_model_consistency(params) < 1:
                    params = self.make_random_model(params0_dict)
                smpdf = self.SM_PDF(params)
                p0 = smpdf.get_p0()
                llk = self.eval_loglikelihood(p0, self.s_obs, params)
            if while_count < maxcount:
                return llk, smpdf
            else:
                return np.nan, np.nan
        else:
            return np.nan, np.nan

    def test_model_consistency(self, params):
        lnan = len([k for k in params if np.isnan(k)])
        lneg = len([k for k in params if k < 0])
        if self.model_type == 'A' or (self.model_type == 'D'):
            [s_h, s_wilt, s_star, s_fc,
             e_max, f_w,
             rf_alpha, rf_lambda, delta,
             a, b, ks, n, zr, e_max_dry, t_d] = params
            if s_star < s_wilt or \
                    s_fc > 1 or \
                    s_fc < s_star or \
                    s_wilt < s_h or \
                    e_max < 0.01 or \
                    f_w > self.p_ranges['f_w'][1] or \
                    e_max > self.p_ranges['e_max'][1] or \
                    e_max_dry > self.p_ranges['e_max_dry'][1] or \
                    zr > self.p_ranges['Zr'][1] or \
                    lneg > 0 or \
                    lnan > 0:
                test = 0
            else:
                test = 1
        elif self.model_type == 'L':
            [s_h, s_wilt, s_star, s_fc,
             e_max, f_w,
             rf_alpha, rf_lambda, delta,
             a, b, ks, n, zr] = params
            if s_star < s_wilt or \
                    s_fc > 1 or \
                    s_fc < s_star or \
                    s_wilt < s_h or \
                    e_max < 0.01 or \
                    f_w > self.p_ranges['f_w'][1] or \
                    e_max > self.p_ranges['e_max'][1] or \
                    zr > self.p_ranges['Zr'][1] or \
                    lneg > 0 or \
                    lnan > 0:
                test = 0
            else:
                test = 1
        return test

    def eval_logps(self, p0, s_eval, params):
        def __ev(s):
            p = p0[np.int(np.rint(s * l))]
            return np.log(p)

        l = (len(p0) - 1)
        if s_eval != []:
            return [__ev(s) for s in s_eval]
        else:
            return [-np.inf]

    def eval_loglikelihood(self, p0, s_eval, params):
        p = self.eval_logps(p0, s_eval, params)
        return np.sum(p)

    def mcmc_mh_criteria(self, li, theta_i):
        lii, theta_ii = self.eval_mh_model(theta_i)
        if lii > li:
            return [1, lii, theta_ii]
        elif np.random.uniform(0.0, 1.0) < np.exp(lii - li):
            return [1, lii, theta_ii]
        else:
            return [0, li, theta_i]

    def eval_mh_model(self, theta0):
        params = self.make_mh_model(theta0)
        smpdf = self.SM_PDF(params)
        if self.test_model_consistency(params) == 1:
            p0 = smpdf.get_p0()
            llk = self.eval_loglikelihood(p0, self.s_obs, params)
            if llk == 0 or np.isnan(llk):
                llk = -np.inf
        else:
            llk = -np.inf
        return llk, smpdf

    def make_mh_model(self, theta0, w=0.01):
        params = []
        for vi in self.full_param_name_list:
            if vi in self.unknown_params:
                params.append(np.random.normal(getattr(theta0, vi), w * (self.p_ranges[vi][1] - self.p_ranges[vi][0])))
            else:
                params.append(getattr(theta0, vi))
        return params

    def make_random_model(self, params0):
        params = []
        for vi in self.full_param_name_list:
            if vi in self.unknown_params:
                params.append(np.random.uniform(self.p_ranges[vi][0], self.p_ranges[vi][1]))
            else:
                params.append(params0[vi])
        return params


class Processor(object):
    def __init__(self, model_params_estimate, model_type='A'):
        self.model_type = model_type
        self.model_params_estimate = model_params_estimate
        if (model_type == 'A') or (model_type == 'D'):
            self.full_param_name_list = ['s_h', 's_wilt', 's_star', 's_fc',
                                     'e_max', 'f_w',
                                     'rf_alpha', 'rf_lambda', 'delta',
                                     'a', 'b', 'Ks', 'n', 'Zr', 'e_max_dry', 't_d']
        elif model_type=='L':
            self.full_param_name_list = ['s_h', 's_wilt', 's_star', 's_fc',
                                     'e_max', 'f_w',
                                     'rf_alpha', 'rf_lambda', 'delta',
                                     'a', 'b', 'Ks', 'n', 'Zr']

    def get_mcmc_mh_results(self, s_obs, params_dict0, p_ranges, 
                            nbr_sim=20000, num_pl=3, 
                            burnin=1./2., max_num_pl=10,
                            efficiency_lim=[1, 80]):
        pl_results = []
        fail_conv_count = 0
        fail_eff_count = 0
        it = 0
        while (len(pl_results) < num_pl) and (it < max_num_pl):
            it = it + 1
            bf = Inverse_bayesian_fitting(s_obs,
                                          self.model_params_estimate,
                                          p_ranges,
                                          nbr_sim=nbr_sim,
                                          burnin=burnin,
                                          model_type=self.model_type)
            x = bf.mcmc_mh(params_dict0)
            if x[1] != 'max init random':
                result, efficiency = x
                if (efficiency >= efficiency_lim[0]) and (efficiency < efficiency_lim[1]):
                    pl_results.append(x)
                else:
                    fail_eff_count = fail_eff_count + 1
            else:
                print('max init random')
            if len(pl_results) == num_pl:
                pl_results = self.check_int_convergeance(pl_results)
                if len(pl_results) != num_pl:
                    fail_conv_count = fail_conv_count + 1

        return pl_results, it, fail_conv_count, fail_eff_count

    def check_int_convergeance(self, pl_results0, gr_th=1.1):
        pl_results, efficiency = zip(*pl_results0)
        loglikelihood = [zip(*r)[0] for r in pl_results]

        estimated_params = [zip(*zip(*r)[1]) for r in pl_results]
        estimated_params = zip(*estimated_params)

        lk_mean = [np.mean(x) for x in loglikelihood]
        gr_list = []
        for p, est_ in zip(self.model_params_estimate, estimated_params):
            gr = self.gelman_rubin_diagnostic([x for x in est_])
            gr_list.append(gr)

        gr_l = len([gri for gri in gr_list if gri > gr_th])
        if gr_l == 0:
            pl_results_r = pl_results0
        else:
            lk_mean = [np.mean(x) for x in loglikelihood]
            min_i = lk_mean.index(np.min(lk_mean))
            pl_results_r = [pi for ii, pi in enumerate(pl_results0) if ii != min_i]
        return pl_results_r

    def gelman_rubin_diagnostic(self, results):
        k = np.float(len(results))
        n = np.float(len(results[0]))
        means = [np.mean(r) for r in results]
        all_mean = np.mean(means)
        b = n / (k - 1) * \
            np.sum([(mi - all_mean)**2 for mi in means])
        w = 1. / (k * (n-1)) * \
            np.sum([(ri - mi) ** 2 for (result, mi) in zip(results, means) for ri in result])
        return ((w * (n - 1) / n + b / n) / w)**0.5

    def process_raw_results(self, result_dict, pl_results, outfile_format='short'):
        def __nse(obs, mod):
            mo = np.mean(obs)
            a = np.sum([(mi - oi) ** 2 for mi, oi in zip(mod, obs)])
            b = np.sum([(oi - mo) ** 2 for oi in obs])
            return 1 - a / b
        if self.model_type == 'L':
            SM_PDF = SM_L
        elif self.model_type == 'A':
            SM_PDF = SM_A
        elif self.model_type == 'D':
            SM_PDF = SM_D

        pl_results, efficiency = zip(*pl_results)
        result_dict['efficiency_estimates'] = efficiency
        result_dict['efficiency'] = np.nanmean(efficiency)
        loglikelihood = [zip(*r)[0] for r in pl_results]

        estimated_params = [zip(*zip(*r)[1]) for r in pl_results]
        estimated_params = zip(*estimated_params)
        result_dict['loglikelihood_estimates'] = loglikelihood
        result_dict['loglikelihood'] = np.mean([np.mean(llk) for llk in loglikelihood])

        gr = self.gelman_rubin_diagnostic([x for x in loglikelihood])
        result_dict['loglikelihood_grd'] = gr

        for p, est_ in zip(self.model_params_estimate, estimated_params):
            gr = self.gelman_rubin_diagnostic([x for x in est_])
            result_dict['%s_grd' % p] = gr
            est = [y for x in est_ for y in x]
            result_dict['%s' % p] = np.mean(est)
            result_dict['%s_std' % p] = np.std(est)
            result_dict['%s_estimates' % p] = est_

        theta = [result_dict[vi] for vi in self.full_param_name_list]

        smpdf = SM_PDF(theta)
        p_fitted_norm = smpdf.get_p0()
        cdf = np.cumsum(p_fitted_norm)
        f = interp1d(cdf, np.linspace(0, 1, len(p_fitted_norm)))
        random_p = [np.random.uniform(0, 1) for r in range(365)]
        fit_s = np.array(f(random_p))
        (kstat, kstatp) = ks_2samp(result_dict['s_obs'], fit_s)

        cdf_m_n = cdf / np.max(cdf)
        q_obs = [percentileofscore(result_dict['s_obs'], s_obs_i, 'weak') / 100. for s_obs_i in result_dict['s_obs']]
        s_mod = [(np.abs(cdf_m_n - qi)).argmin() / np.float(len(p_fitted_norm) - 1) for qi in q_obs]
        result_dict['NSE_O'] = __nse(result_dict['s_obs'], s_mod)

        s_mod_2 = [(np.abs(cdf_m_n - qi / 365.)).argmin() / np.float(len(p_fitted_norm) - 1) for qi in range(1, 365)]
        obs_l2 = [np.percentile(result_dict['s_obs'], qi / 365. * 100) for qi in range(1, 365)]
        result_dict['NSE'] = __nse(obs_l2, s_mod_2)

        result_dict['ks_stat'] = kstat
        result_dict['ks_stat_p'] = kstatp
        result_dict['model_s_bias'] = (np.mean(fit_s) - np.mean(result_dict['s_obs'])) / np.mean(result_dict['s_obs'])

        if outfile_format == 'short':
            result_dict['s_obs'] = []
            for k in result_dict.keys():
                if k.endswith('estimates'):
                    result_dict[k] = []

        return result_dict


if __name__ == "__main__":
    pass
