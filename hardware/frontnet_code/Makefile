# Makefile
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# io=uart
PMSIS_OS ?= pulp_os
APP = main

APP_SRCS := $(wildcard DORY_network/src/*.c)

ifndef CORE
CORE=4
else
CORE= $(CORE)
endif

APP_CFLAGS += -DNUM_CORES=$(CORE) -DGAP_SDK=1 -IDORY_network/inc -O3 -w -fno-tree-loop-distribute-patterns -flto
APP_LDFLAGS += -lm -flto
# APP_CFLAGS += -DVERBOSE=1


FLASH_FILES = DORY_network/BNReluConvolution0_weights.hex
FLASH_FILES += DORY_network/BNReluConvolution2_weights.hex
FLASH_FILES += DORY_network/BNReluConvolution3_weights.hex
FLASH_FILES += DORY_network/BNReluConvolution4_weights.hex
FLASH_FILES += DORY_network/BNReluConvolution5_weights.hex
FLASH_FILES += DORY_network/BNReluConvolution6_weights.hex
FLASH_FILES += DORY_network/BNReluConvolution7_weights.hex
FLASH_FILES += DORY_network/FullyConnected8_weights.hex
FLASH_FILES += DORY_network/inputs.hex

READFS_FILES := $(FLASH_FILES)
PLPBRIDGE_FLAGS += -f

include $(RULES_DIR)/pmsis_rules.mk
