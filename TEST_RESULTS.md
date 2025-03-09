============================= test session starts ==============================
platform linux -- Python 3.12.1, pytest-8.3.5, pluggy-1.5.0 -- /home/cistudent/.pyenv/versions/3.12.1/bin/python3
cachedir: .pytest_cache
rootdir: /workspaces/mildew-detector
configfile: pytest.ini
plugins: pylama-8.4.1, anyio-4.7.0
collecting ... collected 3 items

tests/test_mildew_detector.py::test_resize_input_image PASSED            [ 33%]
tests/test_mildew_detector.py::test_load_model_and_predict PASSED        [ 66%]
tests/test_mildew_detector.py::test_download_dataframe_as_csv PASSED     [100%]

=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: Type google._upb._message.MessageMapContainer uses PyType_Spec with a metaclass that has custom tp_new. This is deprecated and will no longer be allowed in Python 3.14.

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: Type google._upb._message.ScalarMapContainer uses PyType_Spec with a metaclass that has custom tp_new. This is deprecated and will no longer be allowed in Python 3.14.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 3 passed, 2 warnings in 4.98s =========================
