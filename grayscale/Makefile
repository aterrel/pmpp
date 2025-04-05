CC = nvcc
CFLAGS = -O2 -Xcompiler -Wall -I./include
TARGET = grayscale
SRC = grayscale.cu

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: clean
