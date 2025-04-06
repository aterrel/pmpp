CC = nvcc
CFLAGS = -O2 -Xcompiler -Wall
TARGET = matmul
SRC = matmul.cu

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: clean
