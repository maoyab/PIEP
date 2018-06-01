import numpy as np


'''
REFERENCES
    Laio, F., A. Porporato, L. Ridolfi, and I. Rodriguez-Iturbe (2001),
        Plants in water-controlled ecosystems Active role in hydrologic processes and response to water stress II. Probabilistic soil moisture dynamics,
        Adv. Water Resour., 24(7), 707-723, doi 10.1016/S0309-1708(01)00005-7.
    Dralle, D. N., and S. E. Thompson (2016),
        A minimalistic probabilistic model for soil moisture in seasonlly dry climates,
        Water Resour. Res., 52, 1507 - 1517
    Rodriguez-Iturbe, I., V. K. Gupta, and E. Waymire (1984),
        Scale considerations in the modeling of temporal rainfall,
        Water Resour. Res., 20(11), 1611-1619, doi 10.1029/WR020i011p01611.
'''


class SM_L(object):
    '''
    From Laio et al., 2001
    Model  assumptions
    - daily time scale
    - at a point
    - soil moisture processes are Markovian (no memory)
    - soil is a horizontal layer with constant homogenious characteristics (depth Zr, porosity n)
    - growing season in which climate, vegetation parameters are constant
    - rainfall is a marked Poisson process (lambda, alpha)
    - actual precipitation (Rainfall - Interception) is a censored marked Poisson Process (threshold delta -> lambda_p, alpha_p)
    - between rain events soil moisture loss processes vary within s thresholds: s_fc, s_star, s_w, s_h.

    params definition:
         's_h': relative soil moisture at the hydroscopic point
         's_wilt': relative soil moisture at wilting point
         's_star': relativesoil moisture at incipient stomatal closure
         's_fc': relative soil moisture at field capacity
         'delta': canopy interception of rainfall
         'rf_lambda': mean rainfall frequency
         'rf_alpha': mean rainfall depth
         'Zr': (rooting) soil depth
         'a': a, water retention curve (we dont really care about a except to convert s into potential )
         'b': b, water retention curve (Clapp and Hornberger [1978])
         'Ks': saturated soil hydraulic conductivity (Rawls et al. [1982])
         'n': porosity
         'e_max': max ET
         'e_w': E at s = s_w,  e_w = f_w * e_max
    '''

    def __init__(self, params):
        [self.s_h, self.s_wilt, self.s_star,  self.s_fc,
         self.e_max, self.f_w,
         self.rf_alpha, self.rf_lambda, self.delta,
         self.a, self.b, self.Ks, self.n, self.Zr] = params

        if self.s_h == self.s_wilt:
            self.f_w = 0.0001
        self.e_w = self.f_w * self.e_max
        self.lambda_p = self.get_lambda_p()

        self.eta = self.get_eta()
        self.eta_w = self.get_eta_w()
        self.beta = self.get_beta()
        self.m = self.get_m()
        self.gamma = self.get_gamma()

    def get_beta(self):
        return 2. * self.b + 4.

    def get_gamma(self):
        return self.n * self.Zr / (self.rf_alpha - self.delta)

    def get_lambda_p(self):
        return self.rf_lambda * np.exp(- self.delta / self.rf_alpha)

    def get_eta(self, e=None):
        if e is None:
            return self.e_max / (self.n * self.Zr)
        else:
            return e / (self.n * self.Zr)

    def get_eta_w(self, e=None):
        if e is None:
            return self.e_w / (self.n * self.Zr)
        else:
            return e / (self.n * self.Zr)

    def get_m(self):
        try:
            m = self.Ks \
                / ((self.n * self.Zr)
                    * (np.exp(self.beta * (1. - self.s_fc)) - 1.))
            return m
        except:
            return np.nan

    def __rho_s_h(self, s):
        return self.eta_w * ((s - self.s_h) / (self.s_wilt - self.s_h))

    def __rho_s_w(self, s):
        return self.eta_w + (self.eta - self.eta_w) * ((s - self.s_wilt) / (self.s_star - self.s_wilt))

    def __rho_s_star(self, s):
        return self.eta

    def __rho_s_fc(self, s):
        x = self.beta * (s - self.s_fc)
        e = np.exp(x)
        return self.eta + self.m * (e - 1)

    def __rho(self, s):
        if s > self.s_fc:
            return self.__rho_s_fc(s)
        elif (s > self.s_star) and (s <= self.s_fc):
            return self.__rho_s_star(s)
        elif (s > self.s_wilt) and (s <= self.s_star):
            return self.__rho_s_w(s)
        elif (s > self.s_h) and (s <= self.s_wilt):
            return self.__rho_s_h(s)
        else:
            return 0

    def __p_s_h(self, s):
        # s_h < s <= s_wilt
        ch1 = (s - self.s_h) / (self.s_wilt - self.s_h)
        ch2 = (self.lambda_p * (self.s_wilt - self.s_h) / self.eta_w) - 1.
        p = (1. / self.eta_w) * ch1 ** ch2 * np.exp(- self.gamma * s)
        return p

    def __p_s_w(self, s):
        # s_wilt < s <= s_star
        sw1 = 1. + (self.eta / self.eta_w - 1.) * ((s - self.s_wilt) / (self.s_star - self.s_wilt))
        sw2 = self.lambda_p * (self.s_star - self.s_wilt) / (self.eta - self.eta_w) - 1.
        p = (1. / self.eta_w) * sw1 ** sw2 * np.exp(- self.gamma * s)
        return p

    def __p_s_star(self, s):
        # s_star < s <= s_fc
        sst0 = (1. / self.eta)
        sst_e1 = - self.gamma * s + self.lambda_p / self.eta * (s - self.s_star)
        sst1 = np.exp(sst_e1)
        sst_e2 = self.lambda_p * (self.s_star - self.s_wilt) / (self.eta - self.eta_w)
        sst2 = (self.eta / self.eta_w) ** sst_e2
        p = sst0 * sst1 * sst2
        return p

    def __p_s_fc(self, s):
        # s_fc < s <= 1
        fc_e1 = - (self.beta + self.gamma) * s + self.beta * self.s_fc
        fc_e2 = (self.lambda_p / (self.beta * (self.eta - self.m))) + 1.
        fc_e3 = self.lambda_p * (self.s_star - self.s_wilt) / (self.eta - self.eta_w)

        sfc0 = 1. / self.eta
        sfc1 = np.exp(fc_e1)
        sfc20 = (self.eta * (np.exp(self.beta * s))) / ((self.eta - self.m) * \
                 (np.exp(self.beta * self.s_fc)) + self.m * (np.exp(self.beta * s)))
        sfc2 = sfc20 ** fc_e2
        sfc3 = (self.eta / self.eta_w) ** fc_e3
        sfc4 = np.exp((self.lambda_p / self.eta) * (self.s_fc - self.s_star))

        p = sfc0 * sfc1 * sfc2 * sfc3 * sfc4
        return p

    def __p0(self, s):
        if s > self.s_fc:
            return self.__p_s_fc(s)
        elif (s > self.s_star) and (s <= self.s_fc):
            return self.__p_s_star(s)
        elif (s > self.s_wilt) and (s <= self.s_star):
            return self.__p_s_w(s)
        elif (s > self.s_h) and (s <= self.s_wilt):
            return self.__p_s_h(s)
        else:
            return 0

    def get_p0(self, s_list_len=100):
        s_list = np.linspace(0., 1., (s_list_len + 1))

        if self.s_fc < 1 \
                and self.s_fc >= self.s_star \
                and self.s_star >= self.s_wilt \
                and self.s_wilt >= self.s_h \
                and np.isnan(self.s_fc) == 0 \
                and np.isnan(self.s_star) == 0 \
                and np.isnan(self.s_wilt) == 0 \
                and np.isnan(self.eta_w) == 0 \
                and np.isnan(self.eta) == 0 \
                and self.eta_w > 0:
            p0 = np.array([self.__p0(s) for s in s_list])
            c = np.sum(p0)
            return p0 / c
        else:
            return [0 for s in s_list]


class SM_D(object):

    '''
    modified from Dralle, D. N., and S. E. Thompson (2016)
    Model  assumptions
    t_d >> 1/lambda: length of dry season constant yuear to year
    p_w from SM_L
    p_d: negligeable rainfall: exponential decay following s0
    soil moisture thresholds constant within the year
    e_max_dry different from e_max
    s0: random variable, start of dry season
    '''

    def __init__(self, params):
        [self.s_h, self.s_wilt, self.s_star, self.s_fc,
         self.e_max, self.f_w,
         self.rf_alpha, self.rf_lambda, self.delta,
         self.a, self.b, self.Ks, self.n, self.Zr, 
         self.e_max_dry, self.t_d] = params

        self.smpdf_w = SM_L(params[:-2])
        if self.s_h == self.s_wilt:
            self.f_w = 0.0001

        self.eta_dry = self.smpdf_w.get_eta(e=self.e_max_dry)
        self.eta_w_dry = self.smpdf_w.get_eta_w(e=self.e_max_dry * self.f_w)
        self.beta = self.smpdf_w.get_beta()
        self.m = self.smpdf_w.get_m()

    def get_p0(self, ps0, s_list_len=100):
        s_list = np.linspace(0., 1., (s_list_len + 1))
        s_fin_list = [self.get_s_t(s0, self.t_d) for s0 in s_list]
        p0d = [self.get_p0_sdi(sdi, s_list, s_fin_list, ps0) if sdi >= self.s_h else 0 for sdi in s_list]
        p0d = p0d / np.sum(p0d)
        return p0d

    def get_ps0(self, alternate='simple', s_list_len=100):
        ps0 = self.smpdf_w.get_p0(s_list_len=s_list_len)
        return ps0

    def get_p0_sdi(self, sdi, s_list, s_fin_list, ps0):
        p0d_si = [self.__p_sd_given_s0(s0i, sdi) * ps0i \
                  for s0i, s_fini, ps0i in zip(s_list, s_fin_list, ps0) \
                    if (sdi <= s0i) and (sdi >= s_fini)]
        return np.sum(p0d_si)

    def __p_sd_given_s0(self, s0, sd):
        if (sd > self.s_h) and (sd <= self.s_wilt):
            return self.__psdso_s_h(sd, s0)
        elif (sd > self.s_wilt) and (sd <= self.s_star):
            return self.__psdso_s_w(sd, s0)
        elif (sd > self.s_star) and (sd <= self.s_fc):
            return self.__psdso_s_star(sd, s0)
        elif sd > self.s_fc:
            return self.__psdso_s_fc(sd, s0)
        else:
            return 0

    def __psdso_s_h(self, sd, s0):
        a = self.s_wilt - self.s_h
        b = (sd - self.s_h) * self.t_d * self.eta_w_dry
        return a / b

    def __psdso_s_w(self, sd, s0):
        a = (self.s_star - self.s_wilt) / (self.t_d * (self.eta_dry - self.eta_w_dry))
        b = self.eta_dry - self.eta_w_dry
        c = b * (sd - self.s_wilt) + self.eta_w_dry * (self.s_star - self.s_wilt)
        return a * b / c

    def __psdso_s_star(self, sd, s0):
        return 1 / (self.t_d * self.eta_dry)

    def __psdso_s_fc(self, sd, s0):
        a = 1 / (self.beta * self.t_d * (self.eta_dry - self.m))
        b = (self.eta_dry - self.m) * np.exp(self.beta * (s0 - sd))
        c = - self.eta_dry + self.m + self.m * np.exp(self.beta * (s0 - self.s_fc))
        return a * (self.beta * b / (b + c ))

    def __t_s_fc(self, s0):
        if s0 > self.s_fc:
            return 1 / (self.beta * (self.m - self.eta_dry)) * \
                   (self.beta * (self.s_fc - s0) \
                    + np.log((self.eta_dry - self.m \
                              + self.m * np.exp(self.beta * (s0 - self.s_fc))) \
                             / self.eta_dry)
                    )
        else:
            return 0

    def __t_s_star(self, t_s_fci, s0):
        if s0 > self.s_star:
            if s0 > self.s_fc:
                s0 = self.s_fc
            return t_s_fci \
                   + (s0 - self.s_star) \
                   / self.eta_dry
        else:
            return 0

    def __t_s_wilt(self, t_s_stari, s0):
        if s0 > self.s_wilt:
            if s0 > self.s_star:
                s0 = self.s_star
            return t_s_stari \
                   + (s0 - self.s_wilt) / (self.eta_dry - self.eta_w_dry) \
                   * np.log(self.eta_dry / self.eta_w_dry)
        else:
            return 0

    def __s_h_t(self, t, t_s_wilti):
        return self.s_h + (self.s_wilt - self.s_h) \
                * np.exp(- self.eta_w_dry / (self.s_wilt - self.s_h) \
                    * (t - t_s_wilti))

    def __s_wilt_t(self, t, t_s_stari):
        return self.s_wilt + (self.s_star - self.s_wilt) \
                    * (self.eta_dry / (self.eta_dry - self.eta_w_dry) \
                    * np.exp(- (self.eta_dry - self.eta_w_dry) / (self.s_star - self.s_wilt) \
                        * (t - t_s_stari)) \
                    - self.eta_dry / (self.eta_dry - self.eta_w_dry))

    def __s_star_t(self, t, t_s_fci):
        return self.s_fc - \
            self.eta_dry * (t - t_s_fci)

    def __s_fc_t(self, t):
        return s0 - 1 / self.beta \
                * np.log((self.eta_dry - self.m \
                    + self.m * np.exp(self.beta * (s0 - self.s_fc)) \
                    * self.m * np.exp(self.beta * (self.eta_dry - self.m) * t)\
                    - self.m * np.exp(self.beta * (s0 - self.s_fc))) \
                    / (self.eta_dry - self.m))

    def get_s_t(self, s0, t):
        t_s_fci = self.__t_s_fc(s0)
        t_s_stari = self.__t_s_star(t_s_fci, s0)
        t_s_wilti = self.__t_s_wilt(t_s_stari, s0)
        if t < t_s_fci:
            return self.__s_fc_t(t)
        elif (t < t_s_stari) and (t >= t_s_fci):
            return self.__s_star_t(t, t_s_fci)
        elif (t < t_s_wilti) and (t >= t_s_stari):
            return self.__s_wilt_t(t, t_s_stari)
        else:
            return self.__s_h_t(t, t_s_wilti)


class SM_A(object):

    def __init__(self, params):
        self.smpdf_w = SM_L(params[:-2])
        self.smpdf_d = SM_D(params)
        [self.s_h, self.s_wilt, self.s_star,  self.s_fc,
         self.e_max, self.f_w,
         self.rf_alpha, self.rf_lambda, self.delta,
         self.a, self.b, self.Ks, self.n, self.Zr, 
         self.e_max_dry, self.t_d] = params

    def get_p0(self, s_list_len=100):
        if self.s_fc < 1 \
                and self.s_fc >= self.s_star \
                and self.s_star >= self.s_wilt \
                and self.s_wilt >= self.s_h \
                and np.isnan(self.s_fc) == 0 \
                and np.isnan(self.s_star) == 0 \
                and np.isnan(self.s_wilt) == 0 \
                and self.e_max > 0:
            if self.t_d > 0:
                p0w = self.smpdf_w.get_p0(s_list_len=s_list_len)
                p0d = self.smpdf_d.get_p0(p0w, s_list_len=s_list_len)
                return (1 - self.t_d / 365.) * np.array(p0w) + \
                       self.t_d / 365. * np.array(p0d)
            else:
                p0w = self.smpdf_w.get_p0(s_list_len=s_list_len)
                return p0w
        else:
            return [0 for s in range(s_list_len + 1)]


class Stochastic_rf_char(object):
    def __init__(self, rf_ts):
        #Rodriguez-Iturbe [1984]
        self.rf_ts = np.array(rf_ts)
        self.mu = self.get_mu()
        self.var = self.get_var()
        self.l = self.get_poisson_lambda()
        self.a = self.get_poisson_alpha()

    def get_mu(self):
        return np.mean(self.rf_ts)

    def get_var(self):
        return np.var(self.rf_ts)

    def get_poisson_lambda(self):
        return 2 * self.mu**2 / self.var

    def get_poisson_alpha(self):
        return self.mu / self.l

    def get_lambda_p(self, delta):
        return self.l * np.exp(- delta / self.a)


if __name__ == "__main__":
    pass