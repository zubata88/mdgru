#!/usr/bin/env bash

echo "Running pre-push hook"

#pytest test/test_model.py

if [ $? -ne 0 ]; then
	echo "Tests must pass before commit!"
	exit 1
fi
