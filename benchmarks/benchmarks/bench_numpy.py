import numpy as np

from .common import Benchmark


class AddBench(Benchmark):
    params = (
        [(256*1024*1024,)],
        ['uint8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_add(self, shape, dtype):
        self.x + self.y


class SubBench(Benchmark):
    params = (
        [(10000,), (100000,), (1000000,), (1000000, 1), (500000, 2), (20000, 100)],
        ['float64', 'int']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_sub(self, shape, dtype):
        self.x - self.y


class MulBench(Benchmark):
    params = (
        [(10000,), (100000,), (1000000,), (1000000, 1), (500000, 2), (20000, 100), (1000, 1000)],
        ['float64', 'float32', 'int', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_mul(self, shape, dtype):
        self.x * self.y


class TruedivBench(Benchmark):
    params = (
        [(10000,), (100000,), (1000000,), (1000000, 1), (500000, 2), (20000, 100), (1000, 1000)],
        ['float64', 'float32', 'int', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_truediv(self, shape, dtype):
        self.x / self.y


class DivBench(Benchmark):
    params = (
        [(10000,), (100000,), (1000000,), (1000000, 1), (500000, 2), (20000, 100), (1000, 1000)],
        ['float64', 'float32', 'int', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_divide(self, shape, dtype):
        np.divide(self.x, self.y)


class ModBench(Benchmark):
    params = (
        [(10000,), (100000,), (1000000,), (1000000, 1), (500000, 2), (20000, 100), (1000, 1000)],
        ['float64', 'float32', 'int', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_mod(self, shape, dtype):
        self.x % self.y


class FloordivBench(Benchmark):
    params = (
        [(5000, 100), (500, 1000), (50, 10000), (5, 100000), (100, 20000), (2500, 100), (250, 1000), (25, 10000), (3, 100000)],
        ['int64'],
        [8],
    )
    param_names = ['shape', 'dtype', 'divisor']
    
    def setup(self, shape, dtype, divisor):
        np.random.seed(42)
        self.x = np.random.randint(-1_000_000, 1_000_000, size=shape, dtype=dtype)
        self.y = np.int64(divisor)
    
    def time_floordiv(self, shape, dtype, divisor):
        self.x // self.y


class PowBench(Benchmark):
    params = (
        [(1000, 1000), (1000, 10000), (100000, 2)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_pow(self, shape, dtype):
        self.x ** self.y


class EqBench(Benchmark):
    params = (
        [(1000, 1000), (10000, 10), (100000, 3), (1000000, 10), (1000000, 3), (10, 4), (100000, 4)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_eq(self, shape, dtype):
        self.x == self.y


class NeBench(Benchmark):
    params = (
        [(1000, 1000), (10000, 10), (100000, 3), (1000000, 10), (10, 4), (100000, 4)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_ne(self, shape, dtype):
        self.x != self.y


class GeBench(Benchmark):
    params = (
        [(1000, 1000), (10000, 10), (100000, 3), (1000000, 10), (10, 4), (100000, 4)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_ge(self, shape, dtype):
        self.x >= self.y


class GtBench(Benchmark):
    params = (
        [(1000, 1000), (20000, 100), (10000, 10), (100000, 3), (1000000, 10), (10, 4), (100000, 4)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_gt(self, shape, dtype):
        self.x > self.y


class LeBench(Benchmark):
    params = (
        [(1000, 1000), (10000, 10), (100000, 3), (1000000, 10), (10, 4), (100000, 4)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_le(self, shape, dtype):
        self.x <= self.y


class LtBench(Benchmark):
    params = (
        [(1000, 1000), (10000, 10), (100000, 3), (1000000, 10), (10, 4), (100000, 4)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(*shape).astype(dtype)
    
    def time_lt(self, shape, dtype):
        self.x < self.y


class AllBench(Benchmark):
    params = (
        [(1000,), (1000000,), (100000,)],
        ['bool', 'int64', 'float64', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_all(self, shape, dtype):
        np.all(self.x)


class AnyBench(Benchmark):
    params = (
        [(1000,), (1000000,), (100000,)],
        ['bool', 'int64', 'float64', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_any(self, shape, dtype):
        np.any(self.x)


class ArgmaxBench(Benchmark):
    params = (
        [(1000,), (1000000,), (100000,)],
        ['int64', 'int32', 'int8', 'float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_argmax(self, shape, dtype):
        np.argmax(self.x)


class ArgsortBench(Benchmark):
    params = (
        [(1000000,), (1000000, 2), (100000,), (10000,), (10, 2), (500, 1000)],
        ['float64', 'int64', 'int16', 'int32']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_argsort(self, shape, dtype):
        np.argsort(self.x)


class AsarrayBench(Benchmark):
    params = (
        [(100000,), (1000000,), (10000,), (10000, 10)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_asarray(self, shape, dtype):
        np.asarray(self.x)


class AstypeBench(Benchmark):
    params = (
        [(10000, 10), (10000, 1000), (100000, 4), (1000000,)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_astype(self, shape, dtype):
        self.x.astype('float64')


class CopyBench(Benchmark):
    params = (
        [(1000000,)],
        # , (1000000, 1), (10000, 1000), (10000, 10), (100000,)],
        ['float64']
        # , 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_copy(self, shape, dtype):
        np.copy(self.x)


class DotBench(Benchmark):
    params = (
        [(100, 100), (1000, 1000)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.y = np.random.rand(shape[1], shape[0]).astype(dtype)
    
    def time_dot(self, shape, dtype):
        np.dot(self.x, self.y)


class FillBench(Benchmark):
    params = (
        [(3000, 10000), (100000, 3), (1000, 10000), (10000, 1000)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_fill(self, shape, dtype):
        self.x.fill(0)


class FlattenBench(Benchmark):
    params = (
        [(100000, 3), (1000000,), (10000,), (10000, 10)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_flatten(self, shape, dtype):
        np.flatten(self.x)


class InvertBench(Benchmark):
    params = (
        [(1000, 10000), (100000,), (10000, 1000), (100, 1000), (100000,)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_invert(self, shape, dtype):
        np.invert(self.x.astype('int64'))


def _make_fp_predicate_input(shape, dtype):
    x = np.random.rand(*shape).astype(dtype)
    flat = x.reshape(-1)
    flat[::257] = np.nan
    flat[1::257] = np.inf
    flat[2::257] = -np.inf
    return x


class IsnanBench(Benchmark):
    params = (
        [(1000, 10000), (100000,), (100, 1000), (10000, 10), (1000, 1000), (100, 10000), (1000000,)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = _make_fp_predicate_input(shape, dtype)
    
    def time_isnan(self, shape, dtype):
        np.isnan(self.x)


class IsinfBench(Benchmark):
    params = (
        [(1000, 10000), (100000,), (100, 1000), (10000, 10), (1000, 1000), (100, 10000), (1000000,)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = _make_fp_predicate_input(shape, dtype)
    
    def time_isinf(self, shape, dtype):
        np.isinf(self.x)


class IsfiniteBench(Benchmark):
    params = (
        [(1000, 10000), (100000,), (100, 1000), (10000, 10), (1000, 1000), (100, 10000), (1000000,)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = _make_fp_predicate_input(shape, dtype)
    
    def time_isfinite(self, shape, dtype):
        np.isfinite(self.x)


class LexsortBench(Benchmark):
    params = (
        [(10000, 2), (10, 2), (100000, 2), (10000, 1000)],
        ['int16', 'float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_lexsort(self, shape, dtype):
        np.lexsort(self.x)


class MinBench(Benchmark):
    params = (
        [(1000,), (1000000,), (100000, 3), (10, 4), (100000, 4)],
        ['int64', 'bool', 'float64', 'int32', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_min(self, shape, dtype):
        np.min(self.x)


class MaxBench(Benchmark):
    params = (
        [(1000,), (1000000,), (100000, 3), (10, 4), (100000, 4)],
        ['int64', 'bool', 'float64', 'int32', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_max(self, shape, dtype):
        np.max(self.x)


class PartitionBench(Benchmark):
    params = (
        [(1000000, 3), (100000, 4), (100000,), (500000,), (1000, 10000), (10000, 1000)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.kth = min(1, max(0, self.x.size - 1))
    
    def time_partition(self, shape, dtype):
        np.partition(self.x, self.kth)


class PutmaskBench(Benchmark):
    params = (
        [(10000, 10), (1000000,), (100000, 3)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.mask = self.x > 0.5
    
    def time_putmask(self, shape, dtype):
        np.putmask(self.x, self.mask, 0)


class ReduceBench(Benchmark):
    params = (
        [(1000000,), (100000,), (1000,), (10, 4), (100000, 4), (100000, 3)],
        ['int64', 'bool', 'float64', 'int32', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_reduce(self, shape, dtype):
        np.add.reduce(self.x)


class RepeatBench(Benchmark):
    params = (
        [(100000, 3), (100000,)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_repeat(self, shape, dtype):
        np.repeat(self.x, 2, axis=0)


class ReshapeBench(Benchmark):
    params = (
        [(10000,), (100000,), (1000000,)],
        ['int64', 'bool']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_reshape(self, shape, dtype):
        self.x.reshape(-1)


class RollBench(Benchmark):
    params = (
        [(500, 10000)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_roll(self, shape, dtype):
        np.roll(self.x, 10)


class RoundBench(Benchmark):
    params = (
        [(10000, 10)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_round(self, shape, dtype):
        np.round(self.x)


class SearchsortedBench(Benchmark):
    params = (
        [(300000,), (500000,), (10000,), (100000,)],
        ['float64', 'float32', 'int16', 'int32', 'int64', 'int8', 'uint16', 'uint32', 'uint64', 'uint8', 'str']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        if dtype == 'str':
            self.x = np.array(['a', 'b', 'c'] * (shape[0] // 3), dtype=object)
        else:
            self.x = np.random.rand(*shape).astype(dtype)
            self.x.sort()
        self.y = self.x[len(self.x)//2]
    
    def time_searchsorted(self, shape, dtype):
        np.searchsorted(self.x, self.y)


class SumBench(Benchmark):
    params = (
        [(1000,), (1000000,), (10, 4), (100000, 4), (100000,), (100000, 3)],
        ['int64', 'bool', 'float64', 'int32', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_sum(self, shape, dtype):
        np.sum(self.x)


class MeanBench(Benchmark):
    params = (
        [(1000,), (1000000,), (10, 4), (100000, 4), (100000, 3)],
        ['int64', 'bool', 'float64', 'int32', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_mean(self, shape, dtype):
        np.mean(self.x)


class MedianBench(Benchmark):
    params = (
        [(1000,), (1000000,), (100000, 3), (10, 4), (100000, 4)],
        ['int64', 'bool', 'float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_median(self, shape, dtype):
        np.median(self.x)


class StdBench(Benchmark):
    params = (
        [(1000,), (1000000,), (100000, 3), (10, 4), (100000, 4)],
        ['int64', 'float64', 'int32', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_std(self, shape, dtype):
        np.std(self.x)


class VarBench(Benchmark):
    params = (
        [(1000,), (1000000,), (100000, 3), (10, 4), (100000, 4)],
        ['int64', 'bool', 'float64', 'int32', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_var(self, shape, dtype):
        np.var(self.x)


class ProdBench(Benchmark):
    params = (
        [(1000,), (1000000,), (10, 4), (100000, 4)],
        ['int64', 'bool', 'float64', 'int32', 'int8']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_prod(self, shape, dtype):
        np.prod(self.x)


class WhereBench(Benchmark):
    params = (
        [(10000, 10), (100000, 3), (1000000,)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
        self.mask = self.x > 0.5
    
    def time_where(self, shape, dtype):
        np.where(self.mask, self.x, 0)


class ClipBench(Benchmark):
    params = (
        [(10000, 10), (100000,)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_clip(self, shape, dtype):
        np.clip(self.x, 0, 10)


class ZerosBench(Benchmark):
    params = (
        [(1000000,)],
        ['float64', 'int64']
    )
    param_names = ['shape', 'dtype']
    
    def time_zeros(self, shape, dtype):
        np.zeros(shape, dtype=dtype)


class EmptyBench(Benchmark):
    params = (
        [(100000,)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def time_empty(self, shape, dtype):
        np.empty(shape, dtype=dtype)


class CountBench(Benchmark):
    params = (
        [(1000, 10000), (1000, 10000), (1000, 10000)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_count(self, shape, dtype):
        np.sum(~np.isnan(self.x))


class IsnullBench(Benchmark):
    params = (
        [(100, 1000), (1000, 1000)],
        ['float64']
    )
    param_names = ['shape', 'dtype']
    
    def setup(self, shape, dtype):
        np.random.seed(42)
        self.x = np.random.rand(*shape).astype(dtype)
    
    def time_isnull(self, shape, dtype):
        np.isnan(self.x)
