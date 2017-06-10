#!/usr/bin/env python
import sys
major_version, minor_version = sys.version_info[:2]
if major_version < 3 or minor_version < 6:
  raise Exception("Python 3.6 or above required to run")