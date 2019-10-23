CXX = g++
CPPFLAGS += -Wall -Wextra `pkg-config --cflags glfw3 ftgl`
CXXFLAGS += -std=c++14 -s -Ofast -mtune=native -march=native -mfpmath=both
LDFLAGS += -lm -pthread `pkg-config --static --libs glfw3 ftgl`

TARGET = main
SRCS = $(wildcard *.cpp)
OBJS = $(notdir $(SRCS:.cpp=.o))
DEPS = $(notdir $(SRCS:.cpp=.d))

.PHONY: all
all: $(TARGET)

-include $(DEPS)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c -MMD -MP $< $(CPPFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJS)
	rm -f $(DEPS)
	rm -f $(TARGET)
