# pandas/1 — the project's own issue report

- issue: https://github.com/pandas-dev/pandas/issues/34295
- resolved: via PR #34294
- fetched: 2026-10-25T12:31:35Z
- fix-commit-date: 2020-05-21T20:53:53Z
- comments-included: 0
- gate-tree-sha256: dbdf302512bbe3af5463c90646c34e493383dd62ba5911a33d3be6c4d5a65c9c
- gate-answer-sha256: 50852538177fc302291cfd17250c29ed1e5e6c56f1df328412902d14255e9f79
- gate-removals: 0
- raw-sha256: 0f9eb3c92d0e158ba612a54b43b3d9ca8dad674370b3fb0a89f165ad97fb7324
- content-sha256: 0f9eb3c92d0e158ba612a54b43b3d9ca8dad674370b3fb0a89f165ad97fb7324
- raw-hints-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- hints-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

<!-- gate -->

{
  "files": [
    "pandas/tests/dtypes/test_dtypes.py"
  ],
  "ok": true,
  "reason": null,
  "removals": [],
  "ungated_prose": [],
  "hints": null
}

<!-- raw -->

# BUG: `is_string_dtype` incorrectly identifies categorical data

- [X] I have checked that this issue has not already been reported.

- [X] I have confirmed this bug exists on the latest version of pandas.

- [X] (optional) I have confirmed this bug exists on the master branch of pandas.

---

```python
s = pd.Series(['a', 'b', 'c'], dtype='category')
pdt.is_string_dtype(s)
>>> True
```

#### Problem description

The current implementation of `is_string_dtype` incorrectly evaluates to True for categorical series.

#### Expected Output

```python
s = pd.Series(['a', 'b', 'c'], dtype='category')
pdt.is_string_dtype(s)
>>> False
```

#### Output of ``pd.show_versions()``

<details>

INSTALLED VERSIONS
------------------
commit           : None
python           : 3.8.0.final.0
python-bits      : 64
OS               : Darwin
OS-release       : 19.4.0
machine          : x86_64
processor        : i386
byteorder        : little
LC_ALL           : None
LANG             : en_US.UTF-8
LOCALE           : en_US.UTF-8

pandas           : 1.0.3
numpy            : 1.17.3
pytz             : 2019.3
dateutil         : 2.8.1
pip              : 19.3.1
setuptools       : 44.0.0.post20200106
Cython           : 0.29.14
pytest           : 5.3.2
hypothesis       : None
sphinx           : 2.3.1
blosc            : None
feather          : None
xlsxwriter       : None
lxml.etree       : None
html5lib         : None
pymysql          : None
psycopg2         : None
jinja2           : 2.10.3
IPython          : 7.11.1
pandas_datareader: None
bs4              : None
bottleneck       : None
fastparquet      : None
gcsfs            : None
lxml.etree       : None
matplotlib       : 3.1.2
numexpr          : None
odfpy            : None
openpyxl         : None
pandas_gbq       : None
pyarrow          : None
pytables         : None
pytest           : 5.3.2
pyxlsb           : None
s3fs             : None
scipy            : 1.4.1
sqlalchemy       : None
tables           : None
tabulate         : None
xarray           : None
xlrd             : None
xlwt             : None
xlsxwriter       : None
numba            : 0.48.0

</details>

<!-- content -->

# BUG: `is_string_dtype` incorrectly identifies categorical data

- [X] I have checked that this issue has not already been reported.

- [X] I have confirmed this bug exists on the latest version of pandas.

- [X] (optional) I have confirmed this bug exists on the master branch of pandas.

---

```python
s = pd.Series(['a', 'b', 'c'], dtype='category')
pdt.is_string_dtype(s)
>>> True
```

#### Problem description

The current implementation of `is_string_dtype` incorrectly evaluates to True for categorical series.

#### Expected Output

```python
s = pd.Series(['a', 'b', 'c'], dtype='category')
pdt.is_string_dtype(s)
>>> False
```

#### Output of ``pd.show_versions()``

<details>

INSTALLED VERSIONS
------------------
commit           : None
python           : 3.8.0.final.0
python-bits      : 64
OS               : Darwin
OS-release       : 19.4.0
machine          : x86_64
processor        : i386
byteorder        : little
LC_ALL           : None
LANG             : en_US.UTF-8
LOCALE           : en_US.UTF-8

pandas           : 1.0.3
numpy            : 1.17.3
pytz             : 2019.3
dateutil         : 2.8.1
pip              : 19.3.1
setuptools       : 44.0.0.post20200106
Cython           : 0.29.14
pytest           : 5.3.2
hypothesis       : None
sphinx           : 2.3.1
blosc            : None
feather          : None
xlsxwriter       : None
lxml.etree       : None
html5lib         : None
pymysql          : None
psycopg2         : None
jinja2           : 2.10.3
IPython          : 7.11.1
pandas_datareader: None
bs4              : None
bottleneck       : None
fastparquet      : None
gcsfs            : None
lxml.etree       : None
matplotlib       : 3.1.2
numexpr          : None
odfpy            : None
openpyxl         : None
pandas_gbq       : None
pyarrow          : None
pytables         : None
pytest           : 5.3.2
pyxlsb           : None
s3fs             : None
scipy            : 1.4.1
sqlalchemy       : None
tables           : None
tabulate         : None
xarray           : None
xlrd             : None
xlwt             : None
xlsxwriter       : None
numba            : 0.48.0

</details>

<!-- raw-hints -->


<!-- hints -->


