#!/bin/bash

CURRENT_PATH="`dirname \"$0\"`"

./$CURRENT_PATH/bin/test-integration-methods
./$CURRENT_PATH/bin/test-log-integration-methods

./$CURRENT_PATH/bin/test-minimization-methods