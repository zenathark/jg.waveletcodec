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

    def __init__(self):
        pass

    def compress(self, wavelet, bpp):
        self.wv = wavelet
        self.bit_bucket = bpp * self.wv.shape[0] * self.wv.shape[1]
        self.initialization()
        wise_bit = self.n
        #sorting
        try:
            while self.n > 0:
                print self.n

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
        except EOFError as e:
            print type(e)
            return [self.wv.shape[0], self.wv.shape[1], self.wv.level,
                    wise_bit, self.wv.filter, self.output]
        return [self.wv.cols, self.wv.rows, self.wv.level,
                wise_bit, self.wv.filter, self.output]

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

    def S_n(self, S):
        if len(S) == 0:
            return False
        T = np.array([i.tolist() for i in S])
        return int((abs(self.wv[T[:, 0], T[:, 1]]).max() >= 2 ** self.n))

    def ProcessS(self, S):
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
        sn = self.S_n(self.I)
        self.out(sn)
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
                            filter=filter_)
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

    def expand(self, stream, width, height, level, wise_bit, filter_):
        self.iInitialization(width, height, level, wise_bit, filter_)
        self.output = stream
        #sorting
        try:
            while self.n > 0:
                print self.n

                for l in list(self.LIS):
                    l = self.LIS.pop()
                    self.iProcessS(l)
                self.iProcessI()
                self.iRefinement()
                self.n -= 1
                self.LIS += self.nextLIS
                self.LSP += self.nextLSP
                self.nextLSP = []
                self.nextLSP = []
                self.LIS.sort(reverse=True)
        except EOFError as e:
            print type(e)
            return self.wv
        return self.wv

    def iProcessS(self, S):
        sn = self.read()
        if sn == 1:
            if len(S) == 1:
                s = S.pop()
                sg = self.read()
                self.createCoeff(s, sg)
                self.push(s)
                if S in self.LIS:
                    self.LIS.remove(S)
            else:
                self.iCodeS(S)
        else:
            if S not in self.LIS:
                self.LIS.append(S)

    def iCodeS(self, S):
        O = self.splitList(S)
        for o in O:
            sn = self.read()
            if sn == 1:
                if len(o) == 1:
                    o = o.pop()
                    sg = self.read()
                    self.createCoeff(o[0], sg)
                    self.push(o)
                else:
                    self.iCodeS(o)

            else:
                self.LIS.push(o)
        pass

    def iProcessI(self):
        sn = self.read()
        if sn == 1:
            self.iCodeI()

    def iCodeI(self):
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

    def iRefinement(self, end):
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
            self.dt[coords[0], coords[1]] = (2 ** self.n) * \
                ((sg * 2 - 1) * -1)

    def writeLog(self, method, reason, obj, size, value):
        if self.debug:
            self.log += [method, reason, obj, size, value]


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
        return super(fv_speck, self).compress(self.wv, bpp)

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

    def refinement(self, end):
        print('iRefinement I' + str(len(self.I)) + ' ' + str(len(self.output)))
        c = self.LSP.index
        while c != end:
            i = self.LSP.data[c]
            fv = self.calculate_fovea_w(i)
            if fv >= self.get_current_bpp():
                if self.dt[i[0], i[1]] > 0:
                    coeff = self.dt[i[0], i[1]]
                else:
                    coeff = abs(self.dt[i[0], i[1]])
                if (coeff & 2 ** self.n) > 0:
                    self.out(1)
                else:
                    self.out(0)
            c = (c + 1) % self.LSP.size

    def iRefinement(self, end):
        c = self.LSP.index
        while c != end:
            i = self.LSP.data[c]
            fv = self.calculate_fovea_w(i)
            if fv >= (self.get_dec_bpp()):
                if (self.read()) > 0:
                    if self.dt[i[0], i[1]] > 0:
                        self.dt[i[0], i[1]] |= 2 ** self.n
                    else:
                        self.dt[i[0], i[1]] = (abs(self.dt[i[0], i[1]]) |
                                               2 ** self.n) * -1
            c = (c + 1) % self.LSP.size

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
                aprx_level_r = math.ceil(math.log(self.wv.rows /
                                                  float(ij[0]), 2))
                if aprx_level_r > self.wv.level:
                    aprx_level_r = self.wv.level + 1
            if ij[1] == 0:
                aprx_level_c = self.wv.level + 1
            else:
                aprx_level_c = math.ceil(math.log(self.wv.rows /
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
                y += float(self.wv.rows) / 2 ** aprx_level
            if aprx_level_c == aprx_level:
                x += float(self.wv.cols) / 2 ** aprx_level
            return (y, x, aprx_level)

    def calculate_fovea_length(self):
        H = self.wv.rows
        W = self.wv.cols
        k = np.zeros(4)
        k[0] = self.norm(self.P[0], H - self.P[1])
        k[1] = self.norm(W - self.P[0], self.P[1])
        k[2] = self.norm(W - self.P[0], H - self.P[1])
        k[3] = self.norm(self.P[0], H - self.P[1])
        self.fovea_length = k.max()

    def printFoveaWindow(self):
        window = np.zeros((self.wv.rows, self.wv.cols))
        points = self.wv.get_z_order(self.wavelet.rows * self.wavelet.cols)
        for i in points:
            window[tuple(i)] = self.calculate_fovea_w(i)
        return window

    def get_current_bpp(self):
        bpp = len(self.output)
        bpp /= float(self.wv.rows * self.wv.cols)
        return bpp

    def get_dec_bpp(self):
        bpp = self._idx
        bpp /= float(self.wv.rows * self.wv.cols)
        return bpp

    def norm(self, x, y):
        mx = abs(x)
        if mx < abs(y):
            mx = abs(y)
        return mx  # math.sqrt(float(x**2 + y ** 2))

    def powerlaw(self, n):
        return self.c * (1 - ((n - self.alpha) / (1 - self.alpha))) \
            ** self.gamma
