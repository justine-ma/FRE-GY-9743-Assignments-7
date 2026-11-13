import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)

class Interpolator1DMonotoneConvex(Interpolator1D):
    """HW7: implement your own monotone, convex IFR interpolator here."""

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        if self.extrap_method_ != ExtrapMethod.FLAT:
            raise ValueError("Interpolator1DMonotoneConvex supports FLAT extrapolation only.")

        self.t_ = np.asarray(self.axis1_, dtype=float)
        self.f_ = np.asarray(self.values_, dtype=float)
        n = self.length_
        if n == 0:
            raise ValueError("At least one node is required.")
        if n > 1 and not np.all(np.diff(self.t_) > 0):
            raise ValueError("Node times must be strictly increasing.")
        if n == 1:
            return  # constant curve, nothing to prepare

        h = np.diff(self.t_)
        s = np.diff(self.f_) / h
        d = self._node_slopes(h, s)

        # Knot position and knot slope for each interval.
        m = n - 1
        self.xi_ = np.empty(m)
        self.dbar_ = np.empty(m)
        for i in range(m):
            dl, dr, si = d[i], d[i + 1], s[i]
            if dl == dr == si:
                lam = 0.5                                  # linear piece
            elif (dl - si) * (dr - si) <= 0.0:
                lam = (dr - si) / (dr - dl)                # convex / concave
            else:
                lam = 0.5                                  # inflection
            self.xi_[i] = self.t_[i] + lam * h[i]
            # Slope at the knot so that the interval's rise equals s_i * h_i.
            self.dbar_[i] = 2.0 * si - lam * dl - (1.0 - lam) * dr

        self.d_ = d
        alpha = self.xi_ - self.t_[:-1]
        beta = self.t_[1:] - self.xi_
        safe_a = np.where(alpha > 0, alpha, 1.0)
        safe_b = np.where(beta > 0, beta, 1.0)
        # f = z + slope * u + c * u^2 on each sub-piece.
        self.cl_ = np.where(alpha > 0, (self.dbar_ - d[:-1]) / (2.0 * safe_a), 0.0)
        self.cr_ = np.where(beta > 0, (d[1:] - self.dbar_) / (2.0 * safe_b), 0.0)
        self.zk_ = self.f_[:-1] + alpha * (d[:-1] + self.dbar_) / 2.0   # value at knot

        # Cumulative integral from t_0 to each node.
        pieces = np.array([self._integral_in_interval(i, self.t_[i + 1]) for i in range(m)])
        self.cum_ = np.concatenate(([0.0], np.cumsum(pieces)))

    @staticmethod
    def _node_slopes(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        n = len(s) + 1
        d = np.zeros(n)
        if n == 2:
            d[:] = s[0]
            return d
        # Interior nodes: weighted harmonic mean (Fritsch-Carlson / Brodlie weights).
        for i in range(1, n - 1):
            if s[i - 1] * s[i] > 0.0:
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                d[i] = (w1 + w2) / (w1 / s[i - 1] + w2 / s[i])
                cap = 2.0 * min(abs(s[i - 1]), abs(s[i]))
                d[i] = np.sign(d[i]) * min(abs(d[i]), cap)
        # End nodes: one-sided three-point estimate, made shape preserving.
        d[0] = Interpolator1DMonotoneConvex._end_slope(h[0], h[1], s[0], s[1])
        d[-1] = Interpolator1DMonotoneConvex._end_slope(h[-1], h[-2], s[-1], s[-2])
        return d

    @staticmethod
    def _end_slope(h0: float, h1: float, s0: float, s1: float) -> float:
        d = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1)
        if d * s0 <= 0.0:
            return 0.0
        return np.sign(d) * min(abs(d), 2.0 * abs(s0))

    def _interval_index(self, x: float) -> int:
        i = int(np.searchsorted(self.t_, x, side='right')) - 1
        return min(max(i, 0), self.length_ - 2)

    def _value_in_interval(self, i: int, x: float) -> float:
        if x <= self.xi_[i]:
            u = x - self.t_[i]
            return self.f_[i] + self.d_[i] * u + self.cl_[i] * u * u
        w = x - self.xi_[i]
        return self.zk_[i] + self.dbar_[i] * w + self.cr_[i] * w * w

    def _integral_in_interval(self, i: int, x: float) -> float:
        """Integral of the curve from t_i to x, for t_i <= x <= t_{i+1}."""
        xi = self.xi_[i]
        u = min(x, xi) - self.t_[i]
        total = self.f_[i] * u + self.d_[i] * u ** 2 / 2.0 + self.cl_[i] * u ** 3 / 3.0
        if x > xi:
            w = x - xi
            total += self.zk_[i] * w + self.dbar_[i] * w ** 2 / 2.0 + self.cr_[i] * w ** 3 / 3.0
        return total

    def _antiderivative(self, x: float) -> float:
        """Integral of the curve from t_0 to x (negative for x < t_0)."""
        if self.length_ == 1 or x <= self.t_[0]:
            return self.f_[0] * (x - self.t_[0])
        if x >= self.t_[-1]:
            return self.cum_[-1] + self.f_[-1] * (x - self.t_[-1])
        i = self._interval_index(x)
        return self.cum_[i] + self._integral_in_interval(i, x)

    def interpolate(self, x: float) -> float:
        if self.length_ == 1 or x <= self.t_[0]:
            return float(self.f_[0])
        if x >= self.t_[-1]:
            return float(self.f_[-1])
        return float(self._value_in_interval(self._interval_index(x), x))

    def integrate(self, start_x: float, end_x: float) -> float:
        return float(self._antiderivative(end_x) - self._antiderivative(start_x))

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
