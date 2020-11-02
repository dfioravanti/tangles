#!/bin/bash

pytest --cov=src test --cov-report term-missing

echo DONE!