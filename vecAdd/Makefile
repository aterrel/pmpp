CC = nvcc
CFLAGS = -O2 -Xcompiler -Wall
TARGET = vecAdd
SRC = vecAdd.cu

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: clean
