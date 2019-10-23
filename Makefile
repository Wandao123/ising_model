CXX = g++
CPPFLAGS += -Wall -Wextra -I/usr/include/freetype2
CXXFLAGS += -std=c++14 -s -O2
LDFLAGS += -lm -lglfw -lGLU -lGL -lftgl -pthread

TARGET = main
SRCS = $(wildcard *.cpp)
OBJS = $(notdir $(SRCS:.cpp=.o))
DEPS = $(notdir $(SRCS:.cpp=.d))

.PHONY: all
all: $(TARGET)

-include $(DEPS)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(OBJS_DIR)/%.o: %.cpp
	@[ -d $(OBJS_DIR) ]
	$(CXX) $(CXXFLAGS) -o $@ -c -MMD -MP $< $(CPPFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJS)
	rm -f $(DEPS)
	rm -f $(TARGET)
