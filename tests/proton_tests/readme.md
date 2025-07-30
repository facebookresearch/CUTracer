Use proton-dev branch in triton repo to run the tests.

```bash
git clone https://github.com/triton-inference-server/triton.git
cd triton
git checkout proton-dev
pip install -e .
```

### Run the tests

```bash
cd tests/proton_tests
python ./vector-add-instrumented.py
```
