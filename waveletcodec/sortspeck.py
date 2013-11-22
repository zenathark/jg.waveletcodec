import waveletcodec.wave as wvt
import math
from numpy.numarray.numerictypes import Int
import numpy as np
import waveletcodec.rastering as raster


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
    log = []
    out_idx = 0
    _idx = 0

    def __init__(self):
        pass

    def compress(self, wavelet, bpp):
        self.wv = wavelet
        self.bit_bucket = bpp * self.wv.shape[0] * self.wv.shape[1]
        self.initialization()
        wise_bit = self.n
        #sorting
        result = {
            'rows': self.wv.shape[0],
            'cols': self.wv.shape[1],
            'level': self.wv.level,
            'wise_bit': wise_bit,
            'filter': self.wv.filter,
        }
        try:
            while self.n > 0:
                print self.n
                print self.out_idx

                for l in list(self.LIS):
                    self.ProcessS(l)
                self.ProcessI()
                self.refinement()
                self.n -= 1
                self.LIS += self.nextLIS
                self.nextLIS = []
                self.LSP += self.nextLSP
                self.nextLSP = []
                self.LIS.sort(reverse=True)
        except EOFError:
            pass
        result['payload'] = self.output
        self.f.close()
        return result

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
        self.output = [0] * self.bit_bucket
        self.out_idx = 0
        self.f = open("encode", "w")

    def S_n(self, S):
        if len(S) == 0:
            return False
        T = np.array([i.tolist() for i in S])
        return int((abs(self.wv[T[:, 0], T[:, 1]]).max() >= 2 ** self.n))

    def ProcessS(self, S):
        self.f.write("0-%d " % self.out_idx)
        sn = self.S_n(S)
        self.out(sn)
        if sn == 1:
            if len(S) == 1:
                s = S.pop()
                self.out(self.sign(s))
                self.push(s)
            else:
                self.CodeS(S)
            if S in self.LIS:
                self.LIS.remove(S)
        else:
            if S not in self.LIS:
                self.nextLIS.append(S)

    def CodeS(self, S):
        self.f.write("1-%d " % self.out_idx)
        O = self.splitList(S)
        for o in O:
            sn = self.S_n(o)
            self.out(sn)
            if sn == 1:
                if len(o) == 1:
                    o = o.pop()
                    self.out(self.sign(o))
                    self.push(o)
                else:
                    self.CodeS(o)
            else:
                self.nextLIS.append(o)
        pass

    def ProcessI(self):
        self.f.write("2-%d " % self.out_idx)
        sn = self.S_n(self.I)
        self.out(sn)
        if sn == 1:
            self.CodeI()

    def CodeI(self):
        self.f.write("3-%d " % self.out_idx)
        part = self.splitList(self.I, self.i_partition_size)
        self.i_partition_size = self.i_partition_size * 4
        for i in range(3):
            self.ProcessS(part[i])
        self.I = part[3]
        self.ProcessI()

    def iInitialization(self, width, height, level, filter_, wise_bit):
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
        self._idx = 0
        self.f = open("decode", "w")

    def expand(self, stream, width, height, level, filter_, wise_bit):
        self.iInitialization(width, height, level, filter_, wise_bit)
        self.output = stream
        #sorting
        try:
            while self.n > 0:
                print self.n
                print self._idx

                for l in list(self.LIS):
                    self.iProcessS(l)
                self.iProcessI()
                self.iRefinement()
                self.n -= 1
                self.LIS += self.nextLIS
                self.LSP += self.nextLSP
                self.nextLSP = []
                self.nextLIS = []
                self.LIS.sort(reverse=True)
        except EOFError:
            pass
        self.f.close()
        return self.wv

    def iProcessS(self, S):
        self.f.write("0-%d " % self._idx)
        sn = self.read()
        if sn == 1:
            if len(S) == 1:
                s = S.pop()
                sg = self.read()
                self.createCoeff(s, sg)
                self.push(s)
            else:
                self.iCodeS(S)
            if S in self.LIS:
                self.LIS.remove(S)
        else:
            if S not in self.LIS:
                self.nextLIS.append(S)

    def iCodeS(self, S):
        self.f.write("1-%d " % self._idx)
        O = self.splitList(S)
        for o in O:
            sn = self.read()
            if sn == 1:
                if len(o) == 1:
                    o = o.pop()
                    sg = self.read()
                    self.createCoeff(o, sg)
                    self.push(o)
                else:
                    self.iCodeS(o)
            else:
                self.nextLIS.append(o)
        pass

    def iProcessI(self):
        self.f.write("2-%d " % self._idx)
        sn = self.read()
        if sn == 1:
            self.iCodeI()

    def iCodeI(self):
        self.f.write("3-%d " % self._idx)
        part = self.splitList(self.I, self.i_partition_size)
        self.i_partition_size = self.i_partition_size * 4
        for i in range(3):
            self.iProcessS(part[i])
        self.I = part[3]
        self.iProcessI()

    def sign(self, S):
        if self.wv[S[0], S[0]] >= 0:
            return 0
        else:
            return 1

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
            self.output[self.out_idx] = data
            self.out_idx += 1
        else:
            raise EOFError

    def read(self):
        if self._idx < len(self.output):
            self._idx += 1
            return self.output[self._idx - 1]
        else:
            raise EOFError

    def refinement(self):
        for i in self.LSP:
            if self.wv[i[0], i[1]] > 0:
                coeff = self.wv[i[0], i[1]]
            else:
                coeff = abs(self.wv[i[0], i[1]])
            if (coeff & 2 ** self.n) > 0:
                self.out(1)
            else:
                self.out(0)

    def iRefinement(self):
        for i in self.LSP:
            if (self.read()) > 0:
                if self.wv[i[0], i[1]] > 0:
                    self.wv[i[0], i[1]] |= 2 ** self.n
                else:
                    self.wv[i[0], i[1]] = (abs(self.wv[i[0], i[1]])
                                           | 2 ** self.n) * -1

    def push(self, data):
        self.nextLSP.append(data)

    def createCoeff(self, coords, sg, wv=None):
        if wv is None:
            self.wv[coords[0], coords[1]] = (2 ** self.n) * \
                ((sg * 2 - 1) * -1)


class fv_speck(speck):

    def __init__(self):
        pass

    def compress(self, wavelet, bpp, lbpp, f_center, alpha, c, gamma):
        self.Lbpp = bpp
        self.lbpp = lbpp
        self.alpha = alpha
        self.wv = wavelet
        self.P = f_center
        self.c = c
        self.gamma = gamma
        self.calculate_fovea_length()
        r = super(fv_speck, self).compress(self.wv, bpp)
        r['Lbpp'] = bpp
        r['lbpp'] = lbpp
        r['f_center'] = f_center
        r['c'] = c
        r['alpha'] = alpha
        r['gamma'] = gamma
        return r

    def expand(self, stream, width, height, level, filter_,  wise_bit, bpp,
               lbpp, f_center, alpha, c, gamma):
        self.Lbpp = bpp
        self.lbpp = lbpp
        self.alpha = alpha
        self.P = f_center
        self.c = c
        self.wv = wvt.WCSet(np.zeros((width, height), dtype=Int), level)
        self.wv.level = level
        self.gamma = gamma
        self.calculate_fovea_length()
        return super(fv_speck, self).expand(stream, width, height, level,
                                            filter_, wise_bit)

    # def refinement(self):
    #     for i in self.LSP:
    #         fv = self.calculate_fovea_w(i)
    #         if fv >= self.get_current_bpp():
    #             if self.wv[i[0], i[1]] > 0:
    #                 coeff = self.wv[i[0], i[1]]
    #             else:
    #                 coeff = abs(self.wv[i[0], i[1]])
    #             if (coeff & 2 ** self.n) > 0:
    #                 self.out(1)
    #             else:
    #                 self.out(0)

    # def iRefinement(self):
    #     for i in self.LSP:
    #         fv = self.calculate_fovea_w(i)
    #         if fv >= (self.get_dec_bpp()):
    #             if (self.read()) > 0:
    #                 if self.wv[i[0], i[1]] > 0:
    #                     self.wv[i[0], i[1]] |= 2 ** self.n
    #                 else:
    #                     self.wv[i[0], i[1]] = (abs(self.wv[i[0], i[1]]) |
    #                                            2 ** self.n) * -1

    def calculate_fovea_w(self, ij):
        try:
            P = self.get_center(ij)
        except NameError:
            return self.Lbpp
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
                aprx_level_c = math.ceil(math.log(self.wv.shape[0] /
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
        points = self.wv.get_z_order(self.wavelet.shape[0] *
                                     self.wavelet.shape[1])
        for i in points:
            window[tuple(i)] = self.calculate_fovea_w(i)
        return window

    def get_current_bpp(self):
        # bpp = len(self.output)
        # bpp /= float(self.wv.shape[0] * self.wv.shape[1])V
        bpp = self.out_idx
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
