# 告诉 Make 用 > 代替 Tab 作为命令行前缀
.RECIPEPREFIX = >

CC      ?= cc
AR      ?= ar
CFLAGS ?= -O0 -g -fPIC -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
LDLIBS  ?= -lm -lpthread

BUILD   := build
SRC_DIR := src
INC_DIR := include

CFLAGS  += -I$(INC_DIR) -I$(SRC_DIR)/util
CFLAGS  += -fvisibility=hidden -DCLB_BUILD_SHARED

SRCS := $(shell find $(SRC_DIR) -name '*.c')
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/obj/%.o,$(SRCS))

LIB_STATIC := $(BUILD)/libcolibri.a
LIB_SHARED := $(BUILD)/libcolibri.so
CLI_BIN    := $(BUILD)/colibri

.PHONY: all clean test bench

all: $(LIB_STATIC) $(LIB_SHARED) $(CLI_BIN)

$(BUILD)/obj/%.o: $(SRC_DIR)/%.c
> @mkdir -p $(dir $@)
> $(CC) $(CFLAGS) -c $< -o $@

$(LIB_STATIC): $(OBJS)
> @mkdir -p $(dir $@)
> $(AR) rcs $@ $^

$(LIB_SHARED): $(OBJS)
> @mkdir -p $(dir $@)
> $(CC) -shared $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(CLI_BIN): cli/main.c $(LIB_STATIC)
> $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_STATIC) $(LDLIBS)

TEST_SRCS := $(wildcard tests/*.c)
TEST_BIN  := $(BUILD)/colibri_tests
$(TEST_BIN): $(TEST_SRCS) $(LIB_STATIC)
> $(CC) $(CFLAGS) -I$(SRC_DIR)/util -o $@ $^ $(LDLIBS)

test: $(TEST_BIN)
> @$(TEST_BIN)

clean:
> rm -rf $(BUILD)
