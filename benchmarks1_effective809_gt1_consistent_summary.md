# benchmarks1 有效809条在 add/del 约束后的自洽组合

- 入选条数：`506`
- 排除条数：`303`
- `Before / After` 几何平均：`1.001218858059`
- 强制加回：`benchmarks_add.txt` 全部入选
- 强制删除：`benchmarks_del.txt` 全部排除
- 口径：保留客户用例 6 个 benchmark 的端到端主体、sort/search 内核补充、逐元素/广播与 FFT 前处理补充、RNG 生成器核心补充，以及 argsort 结果消费的连续 take 与部分 complex 索引重排。
