## Wed Jun 18 09:08:29 UTC 2025
ERROR: usage: pytest [options] [file_or_dir] [file_or_dir] [...]
pytest: error: unrecognized arguments: --cov=pytorch --cov-report=term
  inifile: None
  rootdir: /workspace/computer-vision-services

## Wed Jun 18 09:08:36 UTC 2025

==================================== ERRORS ====================================
____________________ ERROR collecting tests/test_config.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_config.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_config.py:1: in <module>
    import yaml
E   ModuleNotFoundError: No module named 'yaml'
_____________________ ERROR collecting tests/test_infer.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_infer.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_infer.py:3: in <module>
    from pytorch.infer import process_image, process_image_segmentation, process_image_classification
E   ModuleNotFoundError: No module named 'pytorch'
_____________________ ERROR collecting tests/test_utils.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_utils.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_utils.py:2: in <module>
    from pytorch.utils import generate_color_palette
E   ModuleNotFoundError: No module named 'pytorch'
=========================== short test summary info ============================
ERROR tests/test_config.py
ERROR tests/test_infer.py
ERROR tests/test_utils.py
!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!!
3 errors in 1.68s
## Wed Jun 18 09:08:45 UTC 2025

==================================== ERRORS ====================================
____________________ ERROR collecting tests/test_config.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_config.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_config.py:2: in <module>
    from pytorch.main import load_config
E   ModuleNotFoundError: No module named 'pytorch'
_____________________ ERROR collecting tests/test_infer.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_infer.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_infer.py:3: in <module>
    from pytorch.infer import process_image, process_image_segmentation, process_image_classification
E   ModuleNotFoundError: No module named 'pytorch'
_____________________ ERROR collecting tests/test_utils.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_utils.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_utils.py:2: in <module>
    from pytorch.utils import generate_color_palette
E   ModuleNotFoundError: No module named 'pytorch'
=========================== short test summary info ============================
ERROR tests/test_config.py
ERROR tests/test_infer.py
ERROR tests/test_utils.py
!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!!
3 errors in 1.64s
## Wed Jun 18 09:09:17 UTC 2025

==================================== ERRORS ====================================
____________________ ERROR collecting tests/test_config.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_config.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_config.py:2: in <module>
    from pytorch.main import load_config
E   ModuleNotFoundError: No module named 'pytorch'
_____________________ ERROR collecting tests/test_infer.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_infer.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_infer.py:3: in <module>
    from pytorch.infer import process_image, process_image_segmentation, process_image_classification
E   ModuleNotFoundError: No module named 'pytorch'
_____________________ ERROR collecting tests/test_utils.py _____________________
ImportError while importing test module '/workspace/computer-vision-services/tests/test_utils.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/root/.pyenv/versions/3.11.12/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/test_utils.py:2: in <module>
    from pytorch.utils import generate_color_palette
E   ModuleNotFoundError: No module named 'pytorch'
=========================== short test summary info ============================
ERROR tests/test_config.py
ERROR tests/test_infer.py
ERROR tests/test_utils.py
!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!!
3 errors in 1.75s
## Wed Jun 18 09:09:39 UTC 2025
.....                                                                    [100%]
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.11.12-final-0 _______________

Name                  Stmts   Miss  Cover
-----------------------------------------
pytorch/__init__.py       0      0   100%
pytorch/infer.py        178     98    45%
pytorch/main.py          71     58    18%
pytorch/utils.py          6      0   100%
-----------------------------------------
TOTAL                   255    156    39%
5 passed in 3.20s
