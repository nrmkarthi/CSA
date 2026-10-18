"""Minimal reverse-mode autodiff on numpy (used only to re-implement the SOTA baselines USAD, GDN and
Deep SVDD, because no deep-learning framework is available in the experiment environment)."""
import numpy as np

F = np.float32


class T:
    __slots__ = ('v', 'g', 'parents', 'back', 'req')

    def __init__(self, v, parents=(), back=None, req=False):
        self.v = np.asarray(v, dtype=F); self.g = None; self.parents = parents; self.back = back; self.req = req

    # ------------------------------------------------------------- graph ops
    def _acc(self, g):
        if not self.req:
            return
        g = g.astype(F)
        self.g = g if self.g is None else self.g + g

    def backward(self, g=None):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t.parents:
                build(p)
            topo.append(t)
        build(self)
        self.g = np.ones_like(self.v) if g is None else g
        for t in reversed(topo):
            if t.back is not None and t.g is not None:
                t.back(t.g)

    @staticmethod
    def _unb(g, shape):
        while g.ndim > len(shape):
            g = g.sum(0)
        for i, s in enumerate(shape):
            if s == 1 and g.shape[i] != 1:
                g = g.sum(i, keepdims=True)
        return g

    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v + o.v, (self, o), req=self.req or o.req)
        out.back = lambda g: (self._acc(T._unb(g, self.v.shape)), o._acc(T._unb(g, o.v.shape)))
        return out

    def __sub__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v - o.v, (self, o), req=self.req or o.req)
        out.back = lambda g: (self._acc(T._unb(g, self.v.shape)), o._acc(T._unb(-g, o.v.shape)))
        return out

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v * o.v, (self, o), req=self.req or o.req)
        out.back = lambda g: (self._acc(T._unb(g * o.v, self.v.shape)), o._acc(T._unb(g * self.v, o.v.shape)))
        return out

    def __matmul__(self, o):
        out = T(self.v @ o.v, (self, o), req=self.req or o.req)

        def back(g):
            a, b = self.v, o.v
            if a.ndim == 2 and b.ndim == 2:
                self._acc(g @ b.T); o._acc(a.T @ g)
            else:  # batched (..., n, m) @ (m, k) or (..., n, m) @ (..., m, k)
                self._acc(T._unb(g @ np.swapaxes(b, -1, -2), a.shape))
                gb = np.swapaxes(a, -1, -2) @ g
                o._acc(T._unb(gb, b.shape) if gb.ndim > b.ndim else gb)
        out.back = back
        return out

    def relu(self):
        m = self.v > 0
        out = T(self.v * m, (self,), req=self.req); out.back = lambda g: self._acc(g * m); return out

    def leaky(self, a=0.2):
        m = np.where(self.v > 0, 1.0, a).astype(F)
        out = T(self.v * m, (self,), req=self.req); out.back = lambda g: self._acc(g * m); return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-np.clip(self.v, -30, 30)))
        out = T(s, (self,), req=self.req); out.back = lambda g: self._acc(g * s * (1 - s)); return out

    def square(self):
        out = T(self.v ** 2, (self,), req=self.req); out.back = lambda g: self._acc(2 * g * self.v); return out

    def mean(self):
        n = self.v.size
        out = T(self.v.mean(), (self,), req=self.req); out.back = lambda g: self._acc(np.full_like(self.v, g / n)); return out

    def sum(self, axis=None, keepdims=False):
        out = T(self.v.sum(axis=axis, keepdims=keepdims), (self,), req=self.req)

        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.v.shape))
        out.back = back
        return out

    def softmax(self, axis=-1):
        e = np.exp(self.v - self.v.max(axis=axis, keepdims=True)); s = e / e.sum(axis=axis, keepdims=True)
        out = T(s, (self,), req=self.req)
        out.back = lambda g: self._acc(s * (g - (g * s).sum(axis=axis, keepdims=True)))
        return out

    def reshape(self, *shape):
        sh = self.v.shape
        out = T(self.v.reshape(*shape), (self,), req=self.req); out.back = lambda g: self._acc(g.reshape(sh)); return out

    def take(self, idx, axis):
        """Gather along `axis` with an integer index array (result shape replaces that axis by idx.shape)."""
        out = T(np.take(self.v, idx, axis=axis), (self,), req=self.req)

        def back(g):
            z = np.zeros_like(self.v)
            gm = np.moveaxis(g, list(range(axis, axis + idx.ndim)), list(range(idx.ndim)))
            zm = np.moveaxis(z, axis, 0)
            np.add.at(zm, idx, gm)
            self._acc(np.moveaxis(zm, 0, axis))
        out.back = back
        return out


def cat(ts, axis=-1):
    out = T(np.concatenate([t.v for t in ts], axis=axis), tuple(ts), req=any(t.req for t in ts))
    sizes = np.cumsum([t.v.shape[axis] for t in ts])[:-1]

    def back(g):
        for t, gg in zip(ts, np.split(g, sizes, axis=axis)):
            t._acc(gg)
    out.back = back
    return out


def param(shape, rng, scale=None):
    fan_in = shape[0] if len(shape) > 1 else shape[-1]
    s = scale if scale is not None else np.sqrt(2.0 / fan_in)
    return T(rng.normal(0, s, shape), req=True)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p = params; self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [np.zeros_like(p.v) for p in params]; self.s = [np.zeros_like(p.v) for p in params]; self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            if p.g is None:
                continue
            g = p.g + self.wd * p.v
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t); sh = self.s[i] / (1 - self.b2 ** self.t)
            p.v -= (self.lr * mh / (np.sqrt(sh) + self.eps)).astype(F)
            p.g = None

    def zero(self):
        for p in self.p:
            p.g = None


def mlp(x, layers, last_act=None):
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = x.relu()
    if last_act == 'sigmoid':
        x = x.sigmoid()
    return x


def make_mlp(sizes, rng, bias=True):
    return [(param((a, b), rng), param((1, b), rng, 0.0) if bias else T(np.zeros((1, b)))) for a, b in zip(sizes[:-1], sizes[1:])]


def grad_check():
    rng = np.random.default_rng(0)
    W = param((4, 3), rng); x = T(rng.normal(size=(5, 4)))
    def f():
        return ((x @ W).sigmoid().square()).mean()
    y = f(); y.backward(); ga = W.g.copy()
    num = np.zeros_like(W.v); e = 1e-3
    for i in range(4):
        for j in range(3):
            W.v[i, j] += e; a = f().v; W.v[i, j] -= 2 * e; b = f().v; W.v[i, j] += e; num[i, j] = (a - b) / (2 * e)
    return float(np.abs(ga - num).max())


if __name__ == '__main__':
    print('max grad error', grad_check())
