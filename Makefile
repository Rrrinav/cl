CL_DBG_LVL = 0
BIN := bin
MAIN := $(BIN)/main
TEST := $(BIN)/test

.PHONY: all test clean

all: $(MAIN)

$(MAIN): main.cpp
	mkdir -p $(BIN)
	g++ -DCL_DEBUG_LEVEL=$(CL_DBG_LVL) -std=c++23 -o $@ $<

test:
	mkdir -p $(BIN)
	g++ -DCL_DEBUG_LEVEL=$(CL_DBG_LVL) -std=c++23 -o $(TEST) tests/tests.cpp -I.
	$(TEST)

clean:
	rm -f $(MAIN) $(TEST)

