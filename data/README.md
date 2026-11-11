# Dataset preparation

The scripts encode benchmark CSVs into NumPy arrays. Download the data
separately; raw examples and prepared caches are not distributed here.

Use the [RecZoo dataset catalog](https://github.com/reczoo/Datasets) for Criteo,
Avazu, and KKBox. The x4 splits are also hosted at
[Criteo_x4](https://huggingface.co/datasets/reczoo/Criteo_x4/tree/main) and
[Avazu_x4](https://huggingface.co/datasets/reczoo/Avazu_x4/tree/main).
Keep the supplied `train.csv`, `valid.csv`, and `test.csv` splits.

```bash
python -m data.prep_criteo_x4 raw/Criteo_x4 cache/criteo_x4
python -m data.prep_avazu_x4 raw/Avazu_x4 cache/avazu_x4
python -m data.prep_bars_x1 raw/KKBox_x1 cache/kkbox_x1 10
```

For KuaiRec, the source directory must contain `big_matrix.csv`,
`user_features.csv`, `item_categories.csv`, and `item_daily_features.csv`:

```bash
python -m data.prep_kuairec raw/KuaiRec/data prepared/kuairec_csv
python -m data.prep_bars_x1 prepared/kuairec_csv cache/kuairec 2
```

The generic BARS encoder requires a `label` column; the final argument is the
minimum category frequency in training. Every script supports `--help`.

## Output format

| File | Shape and type |
| --- | --- |
| `{train,valid,test}_cats.npy` | `[examples, fields]`, `int32` |
| `{train,valid,test}_y.npy` | `[examples]`, `float32` |
| `meta.json` | Field order, vocabulary sizes, offsets, and split statistics |

Vocabularies are fitted on training data only. Retained tokens are sorted
lexicographically. Each field has a local OOV index of zero; adding its field
offset gives the global embedding index. Missing, rare, and unseen tokens map
to that field's OOV. CSV parsing uses pandas' default missing-value recognition.

## Dataset settings

Avazu drops `id`, converts `YYMMDDHH` into hour, weekday (`Sunday=0`), and
weekend indicators, and uses minimum category count two. KKBox keeps composite
values such as `genre_ids`, `composer`, and `lyricist` as single tokens.

KuaiRec labels `watch_ratio > 2` as positive, removes rows with missing `time`
or `date`, and uses a random 8:1:1 split with NumPy seed 2024. Its 36 features
include user/video IDs, hour, weekday, user attributes, item tags as one token,
and static item attributes from the earliest dated item record. Watch ratio and
play statistics are excluded from inputs.

These are statistics of the caches used for the saved experiments:

| Dataset | Fields | Vocabulary, including OOVs | Min. count | Train / validation / test |
| --- | ---: | ---: | ---: | --- |
| Criteo_x4_001 | 39 | 910,708 | 10 | 36,672,493 / 4,584,062 / 4,584,062 |
| Avazu_x4_001 | 24 | 3,750,975 | 2 | 32,343,172 / 4,042,897 / 4,042,898 |
| KKBox_x1 | 19 | 233,909 | 10 | 5,901,932 / 737,743 / 737,743 |
| KuaiRec | 36 | 36,427 | 2 | 10,024,644 / 1,253,081 / 1,253,081 |

## Criteo encoder

The Criteo encoder was reconstructed from the metadata of the cache used in the
experiments; it has not been verified row by row against that cache.
For numerical fields `I1`–`I13`, it uses:

```text
token(x) = floor((ln x)^2)  if x > 2
           x              otherwise
```

The logarithm is natural. Missing numerical values become zero, which maps to
OOV. Values one and two remain distinct; negative integers are retained subject
to the frequency threshold. Categorical missing values map to OOV. The field
order is `I1`–`I13`, then `C1`–`C26`, with minimum count ten.

To compare field order, vocabulary sizes, offsets, and total vocabulary with an
existing cache before encoding:

```bash
python -m data.prep_criteo_x4 raw/Criteo_x4 cache/criteo_x4 \
  --reference-meta /path/to/existing/meta.json
```

Criteo uses memory-mapped output and supports `--chunk-size`; the Avazu and
generic BARS encoders assemble each encoded split in memory.
