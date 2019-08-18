#.SUFFIXES: .o .cpp
#
CC = g++
COMPILE_OPTIONS = -std=c++14 -Og -I/usr/include/freetype2
LINK_OPTIONS = -lm -O3 -lglfw -lGLU -lGL -lftgl #-lGLEW
TARGET = main
SRCS = main.cpp ising_model.cpp
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

all: $(TARGET)

-include $(DEPS)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LINK_OPTIONS)

.cpp.o:
	$(CC) -c -MMD -MP $< $(COMPILE_OPTIONS)

clean:
	rm -f $(OBJS)
	rm -f $(DEPS)
	rm -f $(TARGET)
