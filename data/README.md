# Dataset preparation

These commands convert the existing benchmark splits into the NumPy format used
by the experiment runner. Raw datasets and generated caches are not included.
Run the commands from the repository root after installing its requirements.

Obtain the Criteo, Avazu, and KKBox benchmark splits through the
[RecZoo dataset catalog](https://github.com/reczoo/Datasets). The preserved
benchmark instructions point to the public [Criteo_x4 files](https://huggingface.co/datasets/reczoo/Criteo_x4/tree/main)
and [Avazu_x4 files](https://huggingface.co/datasets/reczoo/Avazu_x4/tree/main).
Download and extract the CSV splits before running these encoders; the scripts
do not bypass dataset access terms or automatically download data.

## Commands

Each BARS input directory must contain `train.csv`, `valid.csv`, and `test.csv`.
Use the supplied benchmark splits; these scripts do not resplit Criteo, Avazu,
or KKBox.

```bash
python -m data.prep_criteo_x4 raw/Criteo_x4 cache/criteo_x4
python -m data.prep_avazu_x4 raw/Avazu_x4 cache/avazu_x4
python -m data.prep_bars_x1 raw/KKBox_x1 cache/kkbox_x1 10
```

KuaiRec preparation first creates split CSVs, then encodes their categorical
features. Its source directory must contain `big_matrix.csv`,
`user_features.csv`, `item_categories.csv`, and `item_daily_features.csv`.

```bash
python -m data.prep_kuairec raw/KuaiRec/data prepared/kuairec_csv
python -m data.prep_bars_x1 prepared/kuairec_csv cache/kuairec 2
```

The generic BARS encoder also accepts other CSV datasets with a `label` column.
The final positional argument is the minimum training-set category count.
Every script supports `--help`.

## Cache format

Every encoded directory contains:

- `{train,valid,test}_cats.npy`: `int32`, shaped `[examples, fields]`.
- `{train,valid,test}_y.npy`: `float32`, shaped `[examples]`.
- `meta.json`: field order, vocabulary sizes, global field offsets, and split
  statistics.

Vocabularies are fitted on training data only. Retained tokens are sorted
lexicographically as strings. Each field has its own local OOV index 0, and
adding its field offset produces the global embedding index. There is no extra
padding index. Unseen, insufficiently frequent, and missing tokens map to that
field's OOV index. CSV parsing uses pandas' default missing-value recognition.

Avazu drops `id`, converts the raw `YYMMDDHH` timestamp into hour, weekday
(`Sunday=0`), and weekend indicators, and retains 24 categorical fields with
minimum count 2. The BARS encoder retains composite KKBox features such as
`genre_ids`, `composer`, and `lyricist` as single categorical tokens; it does not
build sequence features.

KuaiRec uses `watch_ratio > 2` as the positive label, drops interactions with
missing `time` or `date`, and performs a random 8:1:1 split with NumPy seed 2024.
Its 36 features include user/video IDs, hour, weekday, categorical user
attributes, item tags as one token, and static item attributes from the earliest
dated item record. Play statistics and watch ratio are not input features.

## Criteo reconstruction and verification scope

The original Criteo preprocessing source was absent from the surviving
experiment directory. `prep_criteo_x4.py` is a reconstruction based on the
surviving cache metadata and sampled CSV-to-cache correspondences, rather than
a recovered copy of that script.

The observed numerical transformation for `I1`--`I13` is:

```text
token(x) = floor((ln x)^2)  if x > 2
           x              otherwise
```

Missing numerical values become zero, and the zero token maps to OOV. Thus 1
and 2 remain distinct, 3 and 4 become 1, and 5 becomes 2. Negative integer values
are retained as tokens subject to the frequency threshold. Categorical missing
values become the empty token and map to OOV. Field order is `I1`--`I13`, then
`C1`--`C26`, with minimum count 10. The logarithm is natural, **not base 2**.
The informal `log2` notation and small-value wording in the benchmark README
do not describe the observed cache mappings precisely; this implementation
follows those mappings.

Checks against the first 20,000 training examples covered all 39 fields. The
transformed tokens had no conflicting observed non-OOV IDs, observed IDs
followed lexicographic token order, and missing/zero tokens mapped to OOV. Small
fixtures additionally checked numerical transformation, training-only
vocabularies, rare/unseen categories, chunked output, and reference-metadata
validation. These checks **do not establish full-dataset or bitwise equivalence**
to the historical experiment cache. No complete vocabulary rebuild was run
as part of this release preparation.

If an existing cache's metadata is available, its field order, per-field
vocabulary sizes, offsets, and total vocabulary can be checked before encoding:

```bash
python -m data.prep_criteo_x4 raw/Criteo_x4 cache/criteo_x4 \
  --reference-meta /path/to/existing/meta.json
```

Matching metadata alone does not verify every encoded row. The Criteo encoder
writes split arrays through memory maps; `--chunk-size` controls CSV chunk size.
The preserved Avazu and generic BARS implementations assemble each encoded
split in memory and require RAM appropriate for the full split.

## Historical cache statistics

These are reference values read from the experiment caches used for the paper,
not statistics from a new full-data preprocessing run.

| Dataset | Fields | Vocabulary including field OOVs | Minimum count | Train / validation / test rows |
| --- | ---: | ---: | ---: | --- |
| Criteo_x4_001 | 39 | 910,708 | 10 | 36,672,493 / 4,584,062 / 4,584,062 |
| Avazu_x4_001 | 24 | 3,750,975 | 2 | 32,343,172 / 4,042,897 / 4,042,898 |
| KKBox_x1 | 19 | 233,909 | 10 | 5,901,932 / 737,743 / 737,743 |
| KuaiRec | 36 | 36,427 | 2 | 10,024,644 / 1,253,081 / 1,253,081 |

The Avazu, generic BARS, and KuaiRec preprocessing bodies are copied from the
experiment scripts with command-line parsing/main entry points added. Small
fixtures passed for all four preparation tools, including Avazu timestamp
features, composite-field tokenization, and KuaiRec labels/static features and
split sizes. The release-preparation environment used NumPy 2.3.5 and pandas
2.2.3. Numerical libraries, dataset versions, and full-cache equivalence remain
part of a complete reproduction check.
