
import waveletcodec.wave as wvt
import math
import numpy as np
import waveletcodec.rastering as raster
import waveletcodec.entropy as tpy


class speck(object):
    #wavelet object
    wv = 0
    LIS = []
    LSP = []
    nextLIS = []
    nextLSP = []
    I = []
    n = 0
    output = []
    i_size_partition = 0
    bit_bucket = 0
    _coding = None

    def __init__(self):
        pass

    def compress(self, wavelet, bpp):
        self.wv = wavelet
        self.bit_bucket = bpp * self.wv.shape[0] * self.wv.shape[1]
        self.initialization()
        wise_bit = self.n
        self._coding = True
        # #sorting
        try:
            while self.n > 0:
                print self.n
                print "%d-%d:%d" % (bpp, self.out_idx, self.bit_bucket)
                for l in list(self.LIS):
                    self.ProcessS(l)
                self.ProcessI()
                self.refinement()
                self.n -= 1
                self.LIS += self.nextLIS
                self.nextLIS = []
                self.LSP += self.nextLSP
                self.nextLSP = []
                # self.LIS.sort(key=lambda X: len(X))  # (reverse=True)
                self.LSP.sort(key=lambda x: raster.get_z_index(x))
        except EOFError:
            pass
            # print type(e)
            # return [self.wv.shape[0], self.wv.shape[1], self.wv.level,
            #         wise_bit, self.wv.filter, self.output]
        r = {}
        r['colums'] = self.wv.shape[1]
        r['rows'] = self.wv.shape[0]
        r['level'] = self.wv.level
        r['wisebit'] = wise_bit
        r['payload'] = self.output
        return r

    def initialization(self):
        X = raster.get_z_order(self.wv.shape[0] * self.wv.shape[1])
        self.LIS = []   # ts.CircularStack(self.wv.cols * self.wv.rows)
        self.nextLIS = []  # ts.CircularStack(self.wv.cols * self.wv.rows)
        self.LSP = []
        self.nextLSP = []
        s_size = (self.wv.shape[0] * self.wv.shape[1] /
                  2 ** (2 * self.wv.level))
        S = set(X[:s_size])
        del X[:s_size]
        self.I = set(X)
        maxs = abs(self.wv)
        self.n = int(math.log(maxs.max(), 2))
        self.LIS.append(S)
        self.i_partition_size = (self.wv.shape[0] / 2 ** self.wv.level) ** 2
        self.output = []
        self.out_idx = 0
        # cheat code
        self.clone = wvt.WCSet(np.zeros((self.wv.shape), dtype=np.int32),
                               self.wv.level, wvt.CDF97)

    def S_n(self, S):
        if len(S) == 0:
            return False
        if self._coding:
            T = np.array([i.tolist() for i in S])
            sn = int((abs(self.wv[T[:, 0], T[:, 1]]).max() >= 2 ** self.n))
            self.out(sn)
            return sn
        else:
            return self.read()

    def ProcessS(self, S):
        sn = self.S_n(S)
        if sn == 1:
            if len(S) == 1:
                s = S.pop()
                self.sign(s)
                self.push(s)
            else:
                self.CodeS(S)
            if S in self.LIS:
                self.LIS.remove(S)
        else:
            if S not in self.LIS:
                self.nextLIS.append(S)

    def CodeS(self, S):
        O = self.splitList(S)
        for o in O:
            sn = self.S_n(o)
            if sn == 1:
                if len(o) == 1:
                    o = o.pop()
                    self.sign(o)
                    self.push(o)
                else:
                    self.CodeS(o)
            else:
                self.nextLIS.append(o)
        pass

    def ProcessI(self):
        sn = self.S_n(self.I)
        if sn == 1:
            self.CodeI()

    def CodeI(self):
        part = self.splitList(self.I, self.i_partition_size)
        self.i_partition_size = self.i_partition_size * 4
        for i in range(3):
            self.ProcessS(part[i])
        self.I = part[3]
        self.ProcessI()

    def iInitialization(self, width, height, level, wise_bit, filter_):
        self.wv = wvt.WCSet(np.zeros((width, height), dtype=Int), level,
                            filter_=filter_)
        self.wv.level = level
        X = raster.get_z_order(self.wv.shape[0] * self.wv.shape[1])
        self.LIS = []
        self.nextLIS = []
        self.LSP = []
        self.nextLSP = []
        s_size = (self.wv.shape[0] * self.wv.shape[1] /
                  2 ** (2 * self.wv.level))
        S = set(X[:s_size])
        del X[:s_size]
        self.I = set(X)
        self.n = wise_bit
        self.LIS.append(S)
        self.i_partition_size = (self.wv.shape[0] / 2 ** self.wv.level) ** 2
        self.out_idx = 0

    def expand(self, stream, width, height, level, wise_bit, filter_):
        self.iInitialization(width, height, level, wise_bit, filter_)
        self.output = stream
        self._coding = False
        #sorting
        try:
            while self.n > 0:
                print self.n
                for l in list(self.LIS):
                    self.ProcessS(l)
                self.ProcessI()
                self.iRefinement()
                self.n -= 1
                self.LIS += self.nextLIS
                self.nextLIS = []
                self.LSP += self.nextLSP
                self.nextLSP = []
                self.LIS.sort(key=lambda X: len(X))  # (reverse=True)
                self.LSP.sort(key=lambda x: raster.get_z_index(x))
        except EOFError:
            pass
        return self.wv

    def sign(self, S):
        if self._coding:
            if self.wv[tuple(S)] >= 0:
                self.out(0)
                self.createCoeff(S, 0, self.clone)
                return 0
            else:
                self.out(1)
                self.createCoeff(S, 1, self.clone)
                return 1
        else:
            self.createCoeff(S, self.read())

    def splitList(self, l, size=0):
        l = list(l)
        if size == 0:
            if len(l) % 4 != 0:
                raise IndexError
            size = int(len(l) / 4)
            return [set(l[i * size:(i + 1) * size]) for i in (0, 1, 2, 3)]
        else:
            return [set(l[i * size:(i + 1) * size]) for i in (0, 1, 2)] \
                + [l[size * 3:]]

    def out(self, data):
        if self.out_idx < self.bit_bucket:
            self.output.append(data)
            self.out_idx += 1
        else:
            raise EOFError

    def read(self):
        if self.out_idx < len(self.output):
            self.out_idx += 1
            return self.output[self.out_idx - 1]
        else:
            raise EOFError

    def refinement(self):
        for i in self.LSP:
            if self.wv[i] > 0:
                coeff = self.wv[i]
                sign = 1
            else:
                coeff = abs(self.wv[i])
                sign = -1
            if (coeff & 2 ** self.n) > 0:
                self.out(1)
                self.clone[i] = (abs(self.clone[i[0], i[1]]) | 2 ** self.n) * sign
            else:
                self.out(0)

    def iRefinement(self):
        for i in self.LSP:
            if (self.read()) > 0:
                if self.wv[i] > 0:
                    self.wv[i] |= 2 ** self.n
                else:
                    self.wv[i] = (abs(self.wv[i])
                                           | 2 ** self.n) * -1

    def push(self, data):
        self.nextLSP.append(tuple(data))

    def createCoeff(self, coords, sg, wv=None):
        sign = 1 if sg == 0 else -1
        if wv is None:
            self.wv[coords] = (2 ** self.n) * sign
        else:
            wv[coords[0], coords[1]] = (2 ** self.n) * sign

    def check(self):
        for i in self.LSP:
            lg2 = int(np.log2(abs(self.clone[i])))
            if (lg2 < self.n):
                print "error"
                raise Exception


class fv_speck(speck):

    def __init__(self):
        pass

    def compress(self, wavelet, bpp, lbpp, f_center, alpha, c, gamma):
        self.Lbpp = bpp
        self.lbpp = lbpp
        self.alpha = alpha
        self.wv = wavelet
        self.dt = wavelet.data
        self.P = f_center
        self.c = c
        self.gamma = gamma
        self.calculate_fovea_length()
        r = super(fv_speck, self).compress(self.wv, bpp)
        r['Lbpp'] = bpp
        r['lbpp'] = lbpp
        r['alpha'] = alpha
        r['center'] = f_center
        r['c'] = c
        r['gamma'] = gamma
        return r

    def expand(self, stream, width, height, level, wise_bit, bpp, lbpp,
               f_center, alpha, c, gamma):
        self.Lbpp = bpp
        self.lbpp = lbpp
        self.alpha = alpha
        self.P = f_center
        self.c = c
        self.wv = wvt.wavelet2D(np.zeros((width, height), dtype=Int), level)
        self.dt = self.wv.data
        self.wv.level = level
        self.gamma = gamma
        self.calculate_fovea_length()
        return super(fv_speck, self).expand(stream, width, height, level,
                                            wise_bit)

    def refinement(self):
        for i in self.LSP:
            fv = self.calculate_fovea_w(i)
            if fv >= self.get_current_bpp():
                if self.wv[i] > 0:
                    coeff = self.wv[i]
                    sign = 1
                else:
                    coeff = abs(self.wv[i])
                    sign = -1
                if (coeff & 2 ** self.n) > 0:
                    self.out(1)
                    self.clone[i] = (abs(self.clone[i]) | 2 ** self.n) * sign
                else:
                    self.out(0)

    def iRefinement(self):
        for i in self.LSP:
            fv = self.calculate_fovea_w(i)
            if fv >= (self.get_dec_bpp()):
                if (self.read()) > 0:
                    if self.wv[i] > 0:
                        self.wv[i] |= 2 ** self.n
                    else:
                        self.wv[i] = (abs(self.wv[i])
                                            | 2 ** self.n) * -1

    def calculate_fovea_w(self, ij):
        try:
            P = self.get_center(ij)
        except NameError:
            return 8 # self.Lbpp
        d = self.norm(P[1] - ij[1], P[0] - ij[0]) * 2 ** P[2] / \
            self.fovea_length
        if d < self.alpha:
            return self.Lbpp
        elif d >= 1:
            return self.lbpp
        else:
            return self.powerlaw(d) * (self.Lbpp - self.lbpp) + self.lbpp

    def get_center(self, ij):
        if (ij[0] == 0 and ij[1] == 0):
            raise NameError("ij on HH")
        else:
            if ij[0] == 0:
                aprx_level_r = self.wv.level + 1
            else:
                aprx_level_r = math.ceil(math.log(self.wv.shape[0] /
                                                  float(ij[0]), 2))
                if aprx_level_r > self.wv.level:
                    aprx_level_r = self.wv.level + 1
            if ij[1] == 0:
                aprx_level_c = self.wv.level + 1
            else:
                aprx_level_c = math.ceil(math.log(self.wv.shape[1] /
                                                  float(ij[1]), 2))
                if aprx_level_c > self.wv.level:
                    aprx_level_c = self.wv.level + 1
            if (aprx_level_r > self.wv.level) and \
               (aprx_level_c > self.wv.level):
#                raise NameError("ij on HH")
                y = float(self.P[0]) / 2 ** (aprx_level_r - 1)
                x = float(self.P[1]) / 2 ** (aprx_level_r - 1)
                return (y, x, aprx_level_r - 1)
            if aprx_level_r <= aprx_level_c:
                aprx_level = aprx_level_r
            else:
                aprx_level = aprx_level_c
            y = float(self.P[0]) / 2 ** aprx_level
            x = float(self.P[1]) / 2 ** aprx_level
            if aprx_level_r == aprx_level:
                y += float(self.wv.shape[0]) / 2 ** aprx_level
            if aprx_level_c == aprx_level:
                x += float(self.wv.shape[1]) / 2 ** aprx_level
            return (y, x, aprx_level)

    def calculate_fovea_length(self):
        H = self.wv.shape[0]
        W = self.wv.shape[1]
        k = np.zeros(4)
        k[0] = self.norm(self.P[0], H - self.P[1])
        k[1] = self.norm(W - self.P[0], self.P[1])
        k[2] = self.norm(W - self.P[0], H - self.P[1])
        k[3] = self.norm(self.P[0], H - self.P[1])
        self.fovea_length = k.max()

    def printFoveaWindow(self):
        window = np.zeros((self.wv.shape[0], self.wv.shape[1]))
        points = self.wv.get_z_order(self.wavelet.shape[0] * self.wavelet.shape[1])
        for i in points:
            window[tuple(i)] = self.calculate_fovea_w(i)
        return window

    def get_current_bpp(self):
        bpp = len(self.output)
        bpp /= float(self.wv.shape[0] * self.wv.shape[1])
        return bpp

    def get_dec_bpp(self):
        bpp = self._idx
        bpp /= float(self.wv.shape[0] * self.wv.shape[1])
        return bpp

    def norm(self, x, y):
        mx = abs(x)
        if mx < abs(y):
            mx = abs(y)
        return mx  # math.sqrt(float(x**2 + y ** 2))

    def powerlaw(self, n):
        return self.c * (1 - ((n - self.alpha) / (1 - self.alpha))) \
            ** self.gamma


class ar_speck(speck):
    _cdc = None

    def __init__(self):
            pass

    def compress(self, wavelet, bpp):
        self._cdc = tpy.abac([0, 1])
        self._cdc._initialize()
        r = super(ar_speck, self).compress(wavelet, bpp)
        r['abac'] = self._cdc.length()
        return r

    def out(self, data):
        if self.out_idx < self.bit_bucket or \
           self._cdc.length() < self.bit_bucket:
            self.output.append(data)
            self._cdc.push(data)
            self.out_idx += 1
        else:
            raise EOFError


class ar_fvspeck(fv_speck):
    _cdc = None

    def __init__(self):
            pass

    def compress(self, wavelet, bpp, lbpp, f_center, alpha, c, gamma):
        self._cdc = tpy.abac([0, 1])
        self._cdc._initialize()
        r = super(ar_fvspeck, self).compress(
            wavelet, bpp, lbpp, f_center, alpha, c, gamma)
        r['abac'] = self._cdc.length()
        return r

    def out(self, data):
        if self.out_idx < self.bit_bucket or \
           self._cdc.length() < self.bit_bucket:
            self.output.append(data)
            self._cdc.push(data)
            self.out_idx += 1
        else:
            raise EOFError

    def get_current_bpp(self):
        bpp = len(self.output)
        bpp /= float(self.wv.shape[0] * self.wv.shape[1])
        if bpp > self.Lbpp:
            return 0
        else:
            return bpp
