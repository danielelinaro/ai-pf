
CC = gcc
#SUNDIALS_DIR := $(shell brew --prefix sundials)
SUNDIALS_DIR=/Users/daniele/.local/sundials
CFLAGS = -g -Wall -O3 -DNDEBUG -I$(SUNDIALS_DIR)/include
LDFLAGS = -g -Wall -O3 -DNDEBUG -L$(SUNDIALS_DIR)/lib -Wl,-rpath,$(SUNDIALS_DIR)/lib
LIBS = -lsundials_ida -lsundials_nvecserial -lsundials_nvecmanyvector -lsundials_core -lm
EXE = SM_with_load

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

SM_with_load: SM_with_load.o
	$(CC) -o $(EXE) SM_with_load.o $(LDFLAGS) $(LIBS) 

clean:
	rm -f $(EXE)
	rm -f *.o
	rm -f *~
